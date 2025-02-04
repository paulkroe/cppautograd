#include "cppgrad.h"
size_t TILE_SIZE = 32;
/**
 * @brief Performs batched tiled matrix multiplication.
 *
 * This function computes the batched matrix multiplication C = A * B using a tiled approach
 * to improve cache efficiency. It assumes A and B are stored in a row-major format and
 * processes multiple batches efficiently by using pointer arithmetic instead of the `[]` operator.
 *
 * @param A Input matrix A, stored as a 1D vector in row-major order.
 * @param B Input matrix B, stored as a 1D vector in row-major order.
 * @param C Output matrix C, stored as a 1D vector in row-major order.
 * @param m Number of rows in A.
 * @param n Shared dimension between A and B.
 * @param p Number of columns in B.
 * @param batch_size Number of batches to process.
 * @param A_offsets Starting offsets of A for each batch.
 * @param B_offsets Starting offsets of B for each batch.
 * @param C_offsets Starting offsets of C for each batch.
 */
void matmul_tiled(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C,
                  size_t m, size_t n, size_t p,
                  size_t batch_size,
                  const std::vector<size_t>& A_offsets, const std::vector<size_t>& B_offsets, const std::vector<size_t>& C_offsets) {

    for (size_t b = 0; b < batch_size; ++b) {
        size_t A_offset = A_offsets[b];
        size_t B_offset = B_offsets[b];
        size_t C_offset = C_offsets[b];

        const float* A_ptr = A.data() + A_offset;
        const float* B_ptr = B.data() + B_offset;
        float* C_ptr = C.data() + C_offset;

        for (size_t i = 0; i < m; i += TILE_SIZE) {
            size_t i_end = std::min(i + TILE_SIZE, m);

            for (size_t j = 0; j < p; j += TILE_SIZE) {
                size_t j_end = std::min(j + TILE_SIZE, p);

                for (size_t k = 0; k < n; k += TILE_SIZE) {
                    size_t k_end = std::min(k + TILE_SIZE, n);

                    /* Compute small tiles */
                    for (size_t ii = i; ii < i_end; ++ii) {
                        float* C_row = C_ptr + ii * p;  // Pointer to C[ii, *]
                        const float* A_row = A_ptr + ii * n;  // Pointer to A[ii, *]

                        for (size_t jj = j; jj < j_end; ++jj) {
                            float sum = 0.0f;
                            float* C_cell = C_row + jj;  // Pointer to C[ii, jj]
                            
                            for (size_t kk = k; kk < k_end; ++kk) {
                                sum += A_row[kk] * B_ptr[kk * p + jj];  // A[ii, kk] * B[kk, jj]
                            }

                            *C_cell += sum;  // C[ii, jj] += sum
                        }
                    }
                }
            }
        }
    }
}

/**
 * @brief Performs tiled matrix multiplication where either A or B may be transposed.
 *
 * This function computes the batched matrix multiplication C = A * B using a tiled approach
 * to improve cache efficiency. It supports cases where matrix A or matrix B (or both) are transposed.
 * Instead of using the `[]` operator for indexing, it leverages direct pointer arithmetic
 * for improved performance.
 *
 * @param A Input matrix A, stored as a 1D vector in row-major order.
 * @param A_shape Shape of matrix A.
 * @param B Input matrix B, stored as a 1D vector in row-major order.
 * @param B_shape Shape of matrix B.
 * @param C Output matrix C, stored as a 1D vector in row-major order.
 * @param C_shape Shape of matrix C.
 * @param batch_shape Shape of the batch dimension.
 * @param A_offsets Starting offsets of A for each batch.
 * @param B_offsets Starting offsets of B for each batch.
 * @param C_offsets Starting offsets of C for each batch.
 * @param m Number of rows in A.
 * @param n Shared dimension between A and B.
 * @param p Number of columns in B.
 * @param transpose_A If true, treats A as transposed.
 * @param transpose_B If true, treats B as transposed.
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

    /* Iterate over batches */
    for (size_t b = 0; b < batch_size; ++b) {
        /* Compute offset into A, B, and C */
        const float* A_ptr = A.data() + A_offsets[b];
        const float* B_ptr = B.data() + B_offsets[b];
        float* C_ptr = C.data() + C_offsets[b];

        /* Process matrix multiplication with tiling */
        for (size_t i = 0; i < m; i += TILE_SIZE) {
            size_t i_end = std::min(i + TILE_SIZE, m);

            for (size_t j = 0; j < p; j += TILE_SIZE) {
                size_t j_end = std::min(j + TILE_SIZE, p);

                for (size_t k = 0; k < n; k += TILE_SIZE) {
                    size_t k_end = std::min(k + TILE_SIZE, n);

                    /* Compute small tiles */
                    for (size_t ii = i; ii < i_end; ++ii) {
                        float* C_row = C_ptr + ii * p;  // Pointer to C[ii, *]

                        for (size_t jj = j; jj < j_end; ++jj) {
                            float sum = 0.0f;
                            float* C_cell = C_row + jj;  // Pointer to C[ii, jj]

                            for (size_t kk = k; kk < k_end; ++kk) {
                                size_t A_index = transpose_A ? (kk * m + ii) : (ii * n + kk);
                                size_t B_index = transpose_B ? (jj * n + kk) : (kk * p + jj);
                                
                                sum += A_ptr[A_index] * B_ptr[B_index];
                            }

                            *C_cell += sum;  // C[ii, jj] += sum
                        }
                    }
                }
            }
        }
    }
}

/**
 * @brief Performs batched matrix multiplication with broadcasting support.
 *
 * This function computes the matrix multiplication C = A.matmul(B), supporting batches and broadcasting.
 * It assumes that A has shape [..., m, n] and B has shape [..., x, p] with n == x, resulting in 
 * an output tensor C of shape [..., m, p]. The operation leverages multithreading to optimize performance.
 *
 * @param other The tensor B to multiply with the current tensor A.
 * @param n_threads Number of threads to utilize for parallel computation.
 * @return Tensor Resulting tensor C of shape [..., m, p].
 */
Tensor Tensor::matmul(const Tensor &other) const {
    
    auto this_shape = this->ptr->shape;
    auto other_shape = other.ptr->shape;

    auto this_data = this->ptr->data;
    auto other_data = other.ptr->data;

    auto this_requires_grad = this->ptr->requires_grad;
    auto other_requires_grad = other.ptr->requires_grad;
    auto result_requires_grad = this->ptr->requires_grad || other.ptr->requires_grad;

    /* ensure that both tensors have at least two dimensions */
    if (this_shape.size() < 2 || other_shape.size() < 2) {
        throw std::invalid_argument(
            "Both tensors must have at least 2 dimensions for matmul."
        );
    }

    /*
     * A in [batch_dims..., m, n]
     * B in [batch_dims..., x, p]
     */
    size_t m = this_shape[this_shape.size() - 2];
    size_t n = this_shape[this_shape.size() - 1];
    size_t x = other_shape[other_shape.size() - 2];
    size_t p = other_shape[other_shape.size() - 1];

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
    if (this_shape.size() > 2 || other_shape.size() > 2) {
        this_batch_shape =
            std::vector<size_t>(this_shape.begin(), this_shape.end() - 2);
        other_batch_shape =
            std::vector<size_t>(other_shape.begin(), other_shape.end() - 2);

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
    
        this_offsets[b] = ravel_index(batch_index, std::vector<size_t>(this_shape.begin(), this_shape.end() - 2)) * m * n;
        other_offsets[b] = ravel_index(batch_index, std::vector<size_t>(other_shape.begin(), other_shape.end() - 2)) * n * p;
        result_offsets[b] = b * (m * p);
    }
    matmul_tiled(this_data, other_data, result_data,
                            m, n, p, batch_size, 
                            this_offsets, other_offsets, result_offsets);
    /* construct backward function */
    Tensor result = Tensor(result_data, result_shape, result_requires_grad);
    /* construct backward function */
    if (result_requires_grad) {

         std::thread::id tid = std::this_thread::get_id();

        /* add result to computation graph */
        {
            std::lock_guard<std::mutex> lock(TensorData::GLOBAL_PARENTS_MUTEX);
            if (this_requires_grad) {
                result.ptr->parents[tid].insert(this->ptr);
            }
        }
        {
            std::lock_guard<std::mutex> lock(TensorData::GLOBAL_PARENTS_MUTEX);
            if (other_requires_grad) {
                result.ptr->parents[tid].insert(other.ptr);
            }
        }
        
        /* Ensure thread-local gradients are initialized */
        std::shared_ptr<TensorData> this_grad, other_grad;
        {
            std::lock_guard<std::mutex> lock(TensorData::GLOBAL_GRAD_MUTEX);
            if (this_requires_grad) {
                if (!this->ptr->thread_gradients[tid]) {
                    this->ptr->thread_gradients[tid] = std::make_shared<TensorData>(std::vector<float>(this_data.size(), 0.0f), this_shape, false);
                }
                this_grad = this->ptr->thread_gradients[tid];
            }
        }
        {
            std::lock_guard<std::mutex> lock(TensorData::GLOBAL_GRAD_MUTEX);
            if (other_requires_grad) {
                if (!other.ptr->thread_gradients[tid]) {
                    other.ptr->thread_gradients[tid] = std::make_shared<TensorData>(std::vector<float>(other_data.size(), 0.0f), other_shape, false);
                }
                other_grad = other.ptr->thread_gradients[tid];
            }
        }

        result.ptr->backward_fn = [this_ptr = this->ptr, other_ptr = other.ptr, result_ptr = result.ptr, batch_shape = batch_shape, m, n, p]() {
       
            std::thread::id tid = std::this_thread::get_id();

            auto this_shape = this_ptr->shape;
            auto other_shape = other_ptr->shape;
            auto result_shape = result_ptr->shape;

            auto this_data = this_ptr->data;
            auto other_data = other_ptr->data;
            auto result_data = result_ptr->data;

            auto this_grad = this_ptr->thread_gradients[tid];
            auto other_grad = other_ptr->thread_gradients[tid];
            auto result_grad = result_ptr->thread_gradients[tid]->data;


            /* compute number of batches */
            size_t batch_size = 1;
            for (auto d : batch_shape) {
                batch_size *= d;
            }

            std::vector<size_t> A_offsets(batch_size, 0);
            std::vector<size_t> B_offsets(batch_size, 0);
            std::vector<size_t> C_offsets(batch_size, 0);

            for (size_t b = 0; b < batch_size; ++b) {
                std::vector<size_t> batch_index = unravel_index(b, batch_shape);
                
                A_offsets[b] = ravel_index(batch_index, std::vector<size_t>(this_shape.begin(), this_shape.end() - 2)) * m * n;
                B_offsets[b] = ravel_index(batch_index, std::vector<size_t>(other_shape.begin(), other_shape.end() - 2)) * n * p;
                C_offsets[b] = b * (m * p);
            }

            /* dA = dC * B^T */
            if (this_ptr->requires_grad && this_grad) {
                matmul_transposed_tiled(result_grad, result_shape, 
                                    other_data, other_shape, 
                                    this_grad->data, this_shape, 
                                    batch_shape,
                                    C_offsets, B_offsets, A_offsets,
                                    m, p, n,
                                    false, true); // No transpose on dC, transpose B
            }

            /* dB = A^T * dC */
            if (other_ptr->requires_grad && other_grad) {
                matmul_transposed_tiled(this_data, this_shape, 
                                    result_grad, result_shape, 
                                    other_grad->data, other_shape, 
                                    batch_shape,
                                    A_offsets, C_offsets, B_offsets,
                                    n, m, p,
                                    true, false); // Transpose A, no transpose on dC
            }


        };
        
    }

    return result;
}