#include "cppgrad.h"
/*
 * Matrix multiplication of two tensors,
 * supporting batches and broadcasting.
 * Assuming 
 * A has shape [..., m, n]
 * B has shape [..., x, p]
 * such that n == x
 * thus:
 * C = A.matul(B) has shape [..., m, p]
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

        /* TODO: this does not account for broadcasting, i.e. [m, n]@[1, p]*/
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

        /* iterate over batches */
        for (size_t b = 0; b < batch_size; ++b) {
            /* convert b to multi-index in the broadcast shape */
            std::vector<size_t> batch_index = unravel_index(b, batch_shape);

            /* figure out which offsets inside A and B this corresponds to */
            size_t A_offset = ravel_index(batch_index, this_batch_shape) * m * n;
            size_t B_offset = ravel_index(batch_index, other_batch_shape) * n * p;
            size_t C_offset = b * (m * p);

            /* matrix multiplication for each batch */
            for (size_t i = 0; i < m; i++) {
                for (size_t j = 0; j < p; j++) {
                    float sum = 0.0f;
                    for (size_t k = 0; k < n; k++) {
                        sum += data[A_offset + i*n + k]
                             * other.data[B_offset + k*p + j];
                    }
                    result_data[C_offset + i*p + j] = sum;
                }
            }
        }

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
                saved_this_shape, saved_other_shape,
                saved_batch_shape, mm, nn, pp
            ]() {
                /* compute number of batches */
                size_t batch_size = 1;
                for (auto d : saved_batch_shape) {
                    batch_size *= d;
                }

                /* dA = dC * B^T */
                if (this_requires_grad && this_grad) {
                    /* iterate over batches */
                    for (size_t b = 0; b < batch_size; b++) {
                        std::vector<size_t> batch_idx =
                            unravel_index(b, saved_batch_shape);

                        /* compute offset into A, used to index into dA */
                        size_t A_offset = ravel_index(batch_idx,
                            std::vector<size_t>(saved_this_shape.begin(),
                                                saved_this_shape.end()-2))
                                          * (mm * nn);

                        /* compute offset into B, used to index into B */
                        size_t B_offset = ravel_index(batch_idx,
                            std::vector<size_t>(saved_other_shape.begin(),
                                                saved_other_shape.end()-2))
                                          * (nn * pp);

                        /* compute offset into C, used to index into dC */
                        size_t C_offset = b * (mm * pp);

                        /* matrix multiplication for each batch */
                        for (size_t i = 0; i < mm; i++) {
                            for (size_t k = 0; k < nn; k++) {
                                float grad_value = 0.0f;
                                for (size_t j = 0; j < pp; j++) {
                                    grad_value += result_grad->data[C_offset + i*pp + j]
                                                * B_data[B_offset + k*pp + j];
                                }
                                /* store result in dA */
                                this_grad->data[A_offset + i*nn + k] += grad_value;
                            }
                        }
                    }
                }

                /* dB = dC * A^T */
                if (other_requires_grad && other_grad) {
                    /* iterate over batches */
                    for (size_t b = 0; b < batch_size; b++) {
                        std::vector<size_t> batch_idx =
                            unravel_index(b, saved_batch_shape);

                        /* compute offset into A, used to index into A */
                        size_t A_offset = ravel_index(batch_idx,
                            std::vector<size_t>(saved_this_shape.begin(),
                                                saved_this_shape.end()-2))
                                          * (mm * nn);

                        /* compute offset into B, used to index into dB */
                        size_t B_offset = ravel_index(batch_idx,
                            std::vector<size_t>(saved_other_shape.begin(),
                                                saved_other_shape.end()-2))
                                          * (nn * pp);

                        /* compute offset into C, used to index into dC */
                        size_t C_offset = b * (mm * pp);

                        /* matrix multiplication for each batch */
                        for (size_t k = 0; k < nn; k++) {
                            for (size_t j = 0; j < pp; j++) {
                                float grad_value = 0.0f;
                                for (size_t i = 0; i < mm; i++) {
                                    grad_value += A_data[A_offset + i*nn + k]
                                                * result_grad->data[C_offset + i*pp + j];
                                }
                                /* store result in dB */
                                other_grad->data[B_offset + k*pp + j] += grad_value;
                            }
                        }
                    }
                }
            };
        }

        return *result;
}
