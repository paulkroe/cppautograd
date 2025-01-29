#include "cppgrad.h"
size_t TILE_SIZE = 32;
/* tiled matrix multiplication */
/* we are using row major order */
/* for efficient memory access it makes more sense to process rows than columns */
void matmul_tile_rows_tiled(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C,
                            size_t m, size_t n, size_t p,
                            size_t batch_size,
                            const std::vector<size_t>& A_offsets, const std::vector<size_t>& B_offsets, const std::vector<size_t>& C_offsets,
                            size_t start_row, size_t end_row) {

    for (size_t b = 0; b < batch_size; ++b) {
        size_t A_offset = A_offsets[b];
        size_t B_offset = B_offsets[b];
        size_t C_offset = C_offsets[b];

        for (size_t i = start_row; i < end_row; i += TILE_SIZE) {
            size_t i_end = std::min(i + TILE_SIZE, end_row);

            for (size_t j = 0; j < p; j += TILE_SIZE) {
                size_t j_end = std::min(j + TILE_SIZE, p);

                for (size_t k = 0; k < n; k += TILE_SIZE) {
                    size_t k_end = std::min(k + TILE_SIZE, n);

                    /* Compute small tiles */
                    for (size_t ii = i; ii < i_end; ++ii) {
                        for (size_t jj = j; jj < j_end; ++jj) {
                            float sum = 0.0f;
                            for (size_t kk = k; kk < k_end; ++kk) {
                                sum += A[A_offset + ii * n + kk] * B[B_offset + kk * p + jj];
                            }
                            C[C_offset + ii * p + jj] += sum;
                        }
                    }
                }
            }
        }
    }
}

void matmul_tile_row_parallel(const std::vector<float>& A, std::vector<size_t> A_shape,
                                    const std::vector<float>& B, std::vector<size_t> B_shape,
                                    std::vector<float>& C,
                                    const size_t batch_size, const std::vector<size_t>& batch_shape,
                                    const size_t m, const size_t n, const size_t p,
                                    const size_t num_threads) {

    std::vector<size_t> A_offsets(batch_size, 0);
    std::vector<size_t> B_offsets(batch_size, 0);
    std::vector<size_t> C_offsets(batch_size, 0);

    for (size_t b = 0; b < batch_size; ++b) {
        std::vector<size_t> batch_index = unravel_index(b, batch_shape);
        
        A_offsets[b] = ravel_index(batch_index, std::vector<size_t>(A_shape.begin(), A_shape.end() - 2)) * m * n;
        B_offsets[b] = ravel_index(batch_index, std::vector<size_t>(B_shape.begin(), B_shape.end() - 2)) * n * p;
        C_offsets[b] = b * (m * p);
    }

    std::vector<std::thread> workers;
    size_t rows_per_thread = m / num_threads;
    size_t remainder = m % num_threads;

    size_t start = 0;
    for (size_t t = 0; t < num_threads; ++t) {
        /* Spread remainder across first threads */
        size_t end = start + rows_per_thread + (t < remainder ? 1 : 0);
        workers.emplace_back(matmul_tile_rows_tiled, std::cref(A), std::cref(B), std::ref(C),
                             m, n, p,
                             batch_size, std::cref(A_offsets), std::cref(B_offsets), std::cref(C_offsets),
                             start, end);
        start = end;
    }

    for (auto& worker : workers) {
        worker.join();
    }
}

void matmul_transposed_tile_row(const std::vector<float>& A, 
                                      const std::vector<size_t>& A_shape,
                                      const std::vector<float>& B, 
                                      const std::vector<size_t>& B_shape,
                                      std::vector<float>& C, 
                                      const std::vector<size_t>& C_shape,
                                      const std::vector<size_t>& batch_shape,
                                      const std::vector<size_t>& A_offsets,
                                      const std::vector<size_t>& B_offsets,
                                      const std::vector<size_t>& C_offsets,
                                      size_t m, size_t n, size_t p, 
                                      bool transpose_A, bool transpose_B, 
                                      size_t start_row, size_t end_row) {
    
    /* Compute batch size */
    size_t batch_size = 1;
    for (auto d : batch_shape) {
        batch_size *= d;
    }

    /* Determine index functions */
    auto get_A_index = transpose_A ?
        [](size_t offset, size_t i, size_t k, size_t m, size_t n) { return offset + k * m + i; }
        :
        [](size_t offset, size_t i, size_t k, size_t m, size_t n) { return offset + i * n + k; };

    auto get_B_index = transpose_B ?
        [](size_t offset, size_t k, size_t j, size_t n, size_t p) { return offset + j * n + k; }
        :
        [](size_t offset, size_t k, size_t j, size_t n, size_t p) { return offset + k * p + j; };

    /* Iterate over batches */
    for (size_t b = 0; b < batch_size; b++) {
        /* Compute offset into A */
        size_t A_offset = A_offsets[b];

        /* Compute offset into B */
        size_t B_offset = B_offsets[b];

        /* Compute offset into C */
        size_t C_offset = C_offsets[b];

        /* Process matrix multiplication with tiling */
        for (size_t i = start_row; i < end_row; i += TILE_SIZE) {
            size_t i_end = std::min(i + TILE_SIZE, end_row);

            for (size_t j = 0; j < p; j += TILE_SIZE) {
                size_t j_end = std::min(j + TILE_SIZE, p);
                for (size_t k = 0; k < n; k += TILE_SIZE) {
                    size_t k_end = std::min(k + TILE_SIZE, n);

                    /* Compute small tiles */
                    for (size_t ii = i; ii < i_end; ++ii) {
                        for (size_t jj = j; jj < j_end; ++jj) {
                            float sum = 0.0f;
                            for (size_t kk = k; kk < k_end; ++kk) {
                                sum += A[get_A_index(A_offset, ii, kk, m, n)]
                                     * B[get_B_index(B_offset, kk, jj, n, p)];
                            }
                            /* Store result in C */
                            C[C_offset + ii * p + jj] += sum;
                        }
                    }
                }
            }
        }
    }
}

void matmul_transposed_tile_row_parallel(const std::vector<float>& A, 
                       const std::vector<size_t>& A_shape,
                       const std::vector<float>& B, 
                       const std::vector<size_t>& B_shape,
                       std::vector<float>& C, 
                       const std::vector<size_t>& C_shape,
                       const std::vector<size_t>& batch_shape,
                       const std::vector<size_t>& A_offsets,
                       const std::vector<size_t>& B_offsets,
                       const std::vector<size_t>& C_offsets,
                       size_t m, size_t n, size_t p, 
                       bool transpose_A, bool transpose_B,
                       const size_t num_threads) {
                        
                    std::vector<std::thread> workers;
                    size_t rows_per_thread = m / num_threads;
                    size_t remainder = m % num_threads;

                    size_t start = 0;
                    for (size_t t = 0; t < num_threads; ++t) {
                        /* Spread remainder across first threads */
                        size_t end = start + rows_per_thread + (t < remainder ? 1 : 0);
                        
                        workers.emplace_back(
                            matmul_transposed_tile_row, 
                            std::cref(A), std::cref(A_shape),
                            std::cref(B), std::cref(B_shape),
                            std::ref(C), std::cref(C_shape),
                            std::cref(batch_shape),
                            std::cref(A_offsets), std::cref(B_offsets), std::cref(C_offsets),
                            m, n, p,
                            transpose_A, transpose_B,
                            start, end
                        );
                        start = end;
                    }

                    for (auto& worker : workers) {
                        worker.join();
                    }
    
}

/*
 * Matrix multiplication of two tensors,
 * supporting batches and broadcasting.
 * Assuming 
 * A has shape [..., m, n]
 * B has shape [..., x, p]
 * such that n == x
 * thus:
 * C = A.matul(B) has shape [..., m, p]
 * n_threads: number of threads to use
 */
Tensor Tensor::matmul(const Tensor &other, const size_t num_threads) const {
    /* ensure that both tensors have at least two dimensions */
    if (shape.size() < 2 || other.shape.size() < 2) {
        throw std::invalid_argument(
            "Both tensors must have at least 2 dimensions for matmul."
        );
    }

    /*
     * A in [batch_dims..., m, n]
     * B in [batch_dims..., x, p]
     */
    size_t m = shape[shape.size() - 2];
    size_t n = shape[shape.size() - 1];
    size_t x = other.shape[other.shape.size() - 2];
    size_t p = other.shape[other.shape.size() - 1];

    if (n != x) {
        throw std::invalid_argument(
            "Inner dimensions of the tensors must match for matmul."
        );
    }

    /* firts determine batch shape */
    std::vector<size_t> this_batch_shape;
    std::vector<size_t> other_batch_shape;
    std::vector<size_t> batch_shape;

    /*
     * if either tensor has extra dims (beyond the last 2),
     * broadcast those leading dims
     */
    if (shape.size() > 2 || other.shape.size() > 2) {
        this_batch_shape =
            std::vector<size_t>(shape.begin(), shape.end() - 2);
        other_batch_shape =
            std::vector<size_t>(other.shape.begin(), other.shape.end() - 2);

        /* broadcast returns the broadcasted shape, empty if shapes are not broadcastable */
        batch_shape = broadcast(this_batch_shape, other_batch_shape);
        if (batch_shape.empty()) {
            printShapes(this_batch_shape, other_batch_shape);
            throw std::invalid_argument("Tensor shapes must be broadcastable for matmul.");
        }
    }

    /* Construct result shape: batch_shape + [m, p] */
    auto result_shape = batch_shape;
    result_shape.push_back(m);
    result_shape.push_back(p);

    /* forward pass */
    size_t batch_size = 1;
    for (auto d : batch_shape) {
        batch_size *= d;
    }

    size_t total_elems = batch_size * m * p;
    /* allocate memory for result data */
    std::vector<float> result_data(total_elems, 0.0f);

    matmul_tile_row_parallel(this->data, this->shape, other.data, other.shape, result_data,
                            batch_size, batch_shape,
                            m, n, p,
                            num_threads);

    /* construct backward function */
    std::shared_ptr<Tensor> result = std::make_shared<Tensor>(result_data, result_shape, requires_grad || other.requires_grad);

    /* construct backward function */
    if (result->requires_grad) {
        /*
        * copy data necessary for backward function
        * to avoid dangling references
        */
        auto this_requires_grad  = requires_grad;
        auto other_requires_grad = other.requires_grad;

        auto this_grad          = this->grad;
        auto other_grad         = other.grad;
        auto result_grad        = result->grad;

        auto A_data             = data;
        auto B_data             = other.data;

        auto this_backward_fn   = this->backward_fn;
        auto other_backward_fn  = other.backward_fn;

        auto saved_this_shape  = shape;
        auto saved_other_shape = other.shape;
        auto saved_result_shape = result_shape;
        auto saved_batch_shape = batch_shape;

        size_t mm = m;
        size_t nn = n;
        size_t pp = p;

        /* insert result into the computation graph */
        if (this_requires_grad)
            result->parents.push_back(std::make_shared<Tensor>(*this));
        if (other_requires_grad)
            result->parents.push_back(std::make_shared<Tensor>(other));

        result->backward_fn = [
            this_requires_grad, other_requires_grad,
            this_grad, other_grad,
            result_grad, A_data, B_data,
            this_backward_fn, other_backward_fn,
            saved_this_shape, saved_other_shape, saved_result_shape,
            saved_batch_shape, mm, nn, pp
        ](const size_t num_threads) {
            /* compute number of batches */
            size_t batch_size = 1;
            for (auto d : saved_batch_shape) {
                batch_size *= d;
            }

            std::vector<size_t> A_offsets(batch_size, 0);
            std::vector<size_t> B_offsets(batch_size, 0);
            std::vector<size_t> C_offsets(batch_size, 0);

            for (size_t b = 0; b < batch_size; ++b) {
                std::vector<size_t> batch_index = unravel_index(b, saved_batch_shape);
                
                A_offsets[b] = ravel_index(batch_index, std::vector<size_t>(saved_this_shape.begin(), saved_this_shape.end() - 2)) * mm * nn;
                B_offsets[b] = ravel_index(batch_index, std::vector<size_t>(saved_other_shape.begin(), saved_other_shape.end() - 2)) * nn * pp;
                C_offsets[b] = b * (mm * pp);
            }
            /* dA = dC * B^T */
            if (this_requires_grad && this_grad) {
                matmul_transposed_tile_row_parallel(result_grad->data, saved_result_shape, 
                                    B_data, saved_other_shape, 
                                    this_grad->data, saved_this_shape, 
                                    saved_batch_shape,
                                    C_offsets, B_offsets, A_offsets,
                                    mm, pp, nn,
                                    false, true, num_threads); // No transpose on dC, transpose B
            }

            /* dB = A^T * dC */
            if (other_requires_grad && other_grad) {
                matmul_transposed_tile_row_parallel(A_data, saved_this_shape, 
                                    result_grad->data, saved_result_shape, 
                                    other_grad->data, saved_other_shape, 
                                    saved_batch_shape,
                                    A_offsets, C_offsets, B_offsets,
                                    nn, mm, pp,
                                    true, false, num_threads); // Transpose A, no transpose on dC
            }
        };

    }

    return *result;
}