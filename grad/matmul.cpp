#include "cppgrad.h"
size_t TILE_SIZE = 32;
/* tiled matrix multiplication */
/* we are using row major order */
/* for efficient memory access it makes more sense to process rows than columns */
void matmul_tiled(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C,
                            size_t m, size_t n, size_t p,
                            size_t batch_size,
                            const std::vector<size_t>& A_offsets, const std::vector<size_t>& B_offsets, const std::vector<size_t>& C_offsets) {

    for (size_t b = 0; b < batch_size; ++b) {
        size_t A_offset = A_offsets[b];
        size_t B_offset = B_offsets[b];
        size_t C_offset = C_offsets[b];

        for (size_t i = 0; i < m; i += TILE_SIZE) {
            size_t i_end = std::min(i + TILE_SIZE, m);

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
/*
 *todo
 */
void matmul_transposed_tiled(const std::vector<float>& A, 
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
                                      bool transpose_A, bool transpose_B) {
    
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
        for (size_t i = 0; i < m; i += TILE_SIZE) {
            size_t i_end = std::min(i + TILE_SIZE, m);

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
Tensor Tensor::matmul(const Tensor &other) const {
    
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

    std::vector<size_t> this_offsets(batch_size, 0);
    std::vector<size_t> other_offsets(batch_size, 0);
    std::vector<size_t> result_offsets(batch_size, 0);

    for (size_t b = 0; b < batch_size; ++b) {
        std::vector<size_t> batch_index = unravel_index(b, batch_shape);
    
        this_offsets[b] = ravel_index(batch_index, std::vector<size_t>(this->shape.begin(), this->shape.end() - 2)) * m * n;
        other_offsets[b] = ravel_index(batch_index, std::vector<size_t>(other.shape.begin(), other.shape.end() - 2)) * n * p;
        result_offsets[b] = b * (m * p);
    }

    matmul_tiled(this->data, other.data, result_data,
                            m, n, p, batch_size, 
                            this_offsets, other_offsets, result_offsets);

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

         std::thread::id tid = std::this_thread::get_id();

        /* Store parents in a thread-safe manner */
        {
            std::lock_guard<std::mutex> lock(GLOBAL_PARENTS_MUTEX);
            if (this_requires_grad) result->parents[tid].insert(std::make_shared<Tensor>(*this));
        }
        {
            std::lock_guard<std::mutex> lock(GLOBAL_PARENTS_MUTEX);
            if (other_requires_grad) result->parents[tid].insert(std::make_shared<Tensor>(other));
        }
        
        /* Ensure thread-local gradients are initialized */
        std::shared_ptr<Tensor> this_grad, other_grad;
        {
            std::lock_guard<std::mutex> lock(GLOBAL_GRAD_MUTEX);
            if (this_requires_grad) {
                if (!this->thread_gradients[tid]) {
                    this->thread_gradients[tid] = std::make_shared<Tensor>(std::vector<float>(this->data.size(), 0.0f), this->shape, false);
                }
                this_grad = this->thread_gradients[tid];
            }
        }
        {
            std::lock_guard<std::mutex> lock(GLOBAL_GRAD_MUTEX);
            if (other_requires_grad) {
                if (!other.thread_gradients[tid]) {
                    other.thread_gradients[tid] = std::make_shared<Tensor>(std::vector<float>(other.data.size(), 0.0f), other.shape, false);
                }
                other_grad = other.thread_gradients[tid];
            }
        }

        result->backward_fn = [
            this_requires_grad, other_requires_grad,
            this_grad, other_grad,
            result, A_data, B_data,
            this_backward_fn, other_backward_fn,
            saved_this_shape, saved_other_shape, saved_result_shape,
            saved_batch_shape, mm, nn, pp
        ]() {

            std::thread::id tid = std::this_thread::get_id();

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
                matmul_transposed_tiled(result->thread_gradients[tid]->data, saved_result_shape, 
                                    B_data, saved_other_shape, 
                                    this_grad->data, saved_this_shape, 
                                    saved_batch_shape,
                                    C_offsets, B_offsets, A_offsets,
                                    mm, pp, nn,
                                    false, true); // No transpose on dC, transpose B
            }

            /* dB = A^T * dC */
            if (other_requires_grad && other_grad) {
                matmul_transposed_tiled(A_data, saved_this_shape, 
                                    result->thread_gradients[tid]->data, saved_result_shape, 
                                    other_grad->data, saved_other_shape, 
                                    saved_batch_shape,
                                    A_offsets, C_offsets, B_offsets,
                                    nn, mm, pp,
                                    true, false); // Transpose A, no transpose on dC
            }
        };

    }

    return *result;
}