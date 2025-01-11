#include "cppgrad.h"
#include <iostream>

void Tensor::backward() {

    if (this->numel(shape) != 1) {
        throw std::runtime_error("Backward only supported for scalar outputs");
    }

    if (!requires_grad) {
        throw std::runtime_error("This tensor does not require gradient");
    }
    // Initialize gradient if necessary
    if (!grad) {
        grad = std::make_shared<Tensor>(std::vector<float>(data.size(), 1.0f), shape, false);
    } else {
        // Ensure the gradient is explicitly set to 1.0 for the target
        std::fill(grad->data.begin(), grad->data.end(), 1.0f);
    }

    if (backward_fn) {
        backward_fn();
    }
}

/* Overloaded + operator */
Tensor Tensor::operator+(const Tensor& other) const{
    std::cout << "Addition operator called" << std::endl;
    if (!shapes_equal(this->shape, other.shape)) {
        throw std::invalid_argument("Tensor shapes must match for addition.");
    }

    // Perform addition
    std::vector<float> result_data(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        result_data[i] = data[i] + other.data[i];
    }

    Tensor result(result_data, shape, requires_grad || other.requires_grad);

    if (result.requires_grad) {
        auto sz = data.size();

        auto this_requires_grad = this->requires_grad;
        auto other_requires_grad = other.requires_grad;

        auto this_grad = this->grad;
        auto other_grad = other.grad;
        auto this_backward_fn = this->backward_fn;
        auto other_backward_fn = other.backward_fn;

        // Capture the result gradients
        auto result_grad = result.grad;

        result.backward_fn = [sz, 
                              this_requires_grad, other_requires_grad,
                              this_grad, other_grad,
                              this_backward_fn, other_backward_fn,
                              result_grad]() {
            // Update this grad
            if (this_requires_grad && this_grad) {
                for (size_t i = 0; i < sz; i++) {
                    this_grad->data[i] += result_grad->data[i];
                }
                if (this_backward_fn) this_backward_fn();
            }

            // Update other grad
            if (other_requires_grad && other_grad) {
                for (size_t i = 0; i < sz; i++) {
                    other_grad->data[i] += result_grad->data[i];
                }
                if (other_backward_fn) other_backward_fn();
            }
        };
    }

    return result;
}

// Unary minus operator
Tensor Tensor::operator-() const {
    /* Negate data */
    std::vector<float> negated_data(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        negated_data[i] = -data[i];
    }
    std::shared_ptr<Tensor> negated_grad = nullptr;

    /* Negated gradient if necessary */
    if (requires_grad) {
        std::vector<float> negated_grad_data(grad->data.size());
        for (size_t i = 0; i < grad->data.size(); i++) {
            negated_grad_data[i] = -grad->data[i];
        }
        negated_grad = std::make_shared<Tensor>(negated_grad_data, grad->requires_grad);
        negated_grad->shape = grad->shape;
    }

    // Create the negated Tensor
    Tensor result(negated_data, shape, requires_grad, negated_grad);

    // Set up the backward function for the unary minus operator
    if (result.requires_grad) {
        auto sz = data.size();            // Size of the tensor
        auto this_requires_grad = this->requires_grad;
        auto this_grad = this->grad;
        auto this_backward_fn = this->backward_fn;
        auto result_grad = result.grad;  // Result's gradient

        result.backward_fn = [sz, this_requires_grad, this_grad, this_backward_fn, result_grad]() {
            if (this_requires_grad && this_grad) {
                // Propagate the negated gradient
                for (size_t i = 0; i < sz; i++) {
                    this_grad->data[i] -= result_grad->data[i];
                }
                // Call the previous backward function if it exists
                if (this_backward_fn) this_backward_fn();
            }
        };
    }

    return result;
}

/* binary minus operator */
Tensor Tensor::operator-(const Tensor& other) const{
    /* check if shapes match */
    if (!shapes_equal(this->shape, other.shape)) {
        throw std::invalid_argument("Tensor shapes must match for subtraction.");
    }

    return *this + (-other);
}

/* Overloaded * operator */
Tensor Tensor::operator*(const Tensor& other) const{
    std::cout << "Multiplication operator called" << std::endl;
    if (!shapes_equal(this->shape, other.shape)) {
        throw std::invalid_argument("Tensor shapes must match for multiplication.");
    }

    // Perform multiplication
    std::vector<float> result_data(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        result_data[i] = data[i] * other.data[i];
    }

    Tensor result(result_data, shape, requires_grad || other.requires_grad);

    if (result.requires_grad) {
        auto sz = data.size();

        auto this_requires_grad = this->requires_grad;
        auto other_requires_grad = other.requires_grad;

        auto this_grad = this->grad;
        auto other_grad = other.grad;
        auto this_backward_fn = this->backward_fn;
        auto other_backward_fn = other.backward_fn;

        auto this_data = this->data;
        auto other_data = other.data;
        auto result_grad = result.grad;

        result.backward_fn = [sz, 
                              this_requires_grad, other_requires_grad,
                              this_grad, other_grad,
                              this_backward_fn, other_backward_fn,
                              this_data, other_data, result_grad]() {
            if (this_requires_grad && this_grad) {
                for (size_t i = 0; i < sz; i++) {
                    this_grad->data[i] += other_data[i] * result_grad->data[i];
                }
                if (this_backward_fn) this_backward_fn();
            }

            if (other_requires_grad && other_grad) {
                for (size_t i = 0; i < sz; i++) {
                    other_grad->data[i] += this_data[i] * result_grad->data[i];
                }
                if (other_backward_fn) other_backward_fn();
            }
        };
    }

    return result;
}

/*
    Matrix multiplication of two tensors, without broadcasting.
    We assume 
    A has shape [b1, b2, ..., bN, m, n]
    B has shape [b1, b2, ..., bN, x, p]
    such that n == x
    thus:
    C = A.matul(B) has shape [b1, b2, ..., bN, m, p]
*/
Tensor Tensor::matmul(const Tensor &other) const{

    /* check matrix dimensions */

    /* num_dim >= 2 for both matrices*/
    if (shape.size() < 2 || other.shape.size() < 2)
        throw std::invalid_argument("Both matrices must have at least 2 columns");

    size_t m = shape[shape.size() - 2];
    size_t n = shape[shape.size() - 1];

    size_t x = other.shape[other.shape.size() - 2];
    size_t p = other.shape[other.shape.size() - 1];

    if (n != x)
        throw std::invalid_argument("Matricies must have matching inner dimensions");
    
    /* check if matrices have the same number of dimensions */
    if (shape.size() != other.shape.size())
        throw std::invalid_argument("Matrices must have the same number of columns, no broadcasting");

    /* check if leading dimensions match */
    std::vector<size_t> batch_shape;
    for (size_t i = 0; i < shape.size() - 2; i++) {
        if (shape[i] != other.shape[i]) {
            throw std::invalid_argument("Batch dimensions must match for bmm (no broadcasting).");
        }
        batch_shape.push_back(shape[i]);
    }

    std::vector<size_t> result_shape = batch_shape;
    result_shape.push_back(m);
    result_shape.push_back(p);

    /* comupte the number of elements */
    size_t total_elems = this->numel(result_shape);
    
    std::vector<float> result_data(total_elems, 0.0f);

    /* forward pass */
    /* triple-nested loop: over each batch, over each i, j, k */

    for (size_t b = 0; b < total_elems/(m*p); b++) {
        // offset in A = b * (m*n)
        // offset in B = b * (n*p)
        // offset in C = b * (m*p)
        size_t A_offset = b * (m * n);
        size_t B_offset = b * (n * p);
        size_t C_offset = b * (m * p);

        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < p; j++) {
                float sum = 0.0f;
                for (size_t k = 0; k < n; k++) {
                    sum += data[A_offset + i*n + k] *
                           other.data[B_offset + k*p + j];
                }
                result_data[C_offset + i*p + j] = sum;
            }
        }
    }

    /* Construct the output tensor */
    bool out_requires_grad = (requires_grad || other.requires_grad);
    Tensor result(result_data, result_shape, out_requires_grad);

    /* Setup the backward_fn if needed */
    if (out_requires_grad) {
        auto this_requires_grad = this->requires_grad;
        auto other_requires_grad = other.requires_grad;

        auto this_grad = this->grad;
        auto other_grad = other.grad;
        auto result_grad = result.grad;

        auto A_data = data;
        auto B_data = other.data;

        auto this_shape = shape;
        auto other_shape = other.shape;
        auto res_shape = result_shape;

        auto this_backward_fn = this->backward_fn;
        auto other_backward_fn = other.backward_fn;

        result.backward_fn = [=]() mutable {
            //
            // We assume:
            //   A has shape: this_shape = [batch..., m, n]
            //   B has shape: other_shape = [batch..., n, p]
            //   C has shape: result_shape = [batch..., m, p]
            //
            // We want: dA = dC * B^T
            //
            if (this_requires_grad && this_grad) {

                // 1. Compute how many batch “slices” we have.
                //    For example, if this_shape = [2, 3, 4, 5], then the last two
                //    dims (4, 5) are (m, n), so the “batch shape” is [2, 3].
                //    The total number of slices = 2 * 3 = 6.
                size_t batch_count = 1;
                for (size_t i = 0; i + 2 < this_shape.size(); i++) {
                    batch_count *= this_shape[i];
                }

                // 2. Extract (m, n) from the last two dimensions
                const size_t m = this_shape[this_shape.size() - 2];
                const size_t n = this_shape[this_shape.size() - 1];

                // 3. p comes from the last dimension of B
                //    (assuming B also has the same batch_count in front,
                //     then shape is [..., n, p]).
                const size_t p = other_shape[other_shape.size() - 1];

                // -------------------------------------------------------
                // NOTE: We do NOT zero out this_grad here.
                // If you need your gradients to start at zero each backward pass,
                // do it outside (e.g. a higher-level 'zero_grad()' call).
                // -------------------------------------------------------

                // 4. Loop over each batch slice
                for (size_t b = 0; b < batch_count; b++) {
                    // Offsets for each slice in the flattened data
                    const size_t offsetA = b * (m * n);
                    const size_t offsetB = b * (n * p);
                    const size_t offsetC = b * (m * p);

                    // 4a. Accumulate gradient w.r.t. A in the b-th slice
                    //     dA[i, k] += sum_{j} of [ dC[i, j] * B[k, j] ]
                    for (size_t i = 0; i < m; i++) {
                        for (size_t k = 0; k < n; k++) {
                            float grad_value = 0.0f;
                            for (size_t j = 0; j < p; j++) {
                                grad_value += result_grad->data[offsetC + i*p + j]
                                            * B_data[offsetB + k*p + j];
                            }
                            this_grad->data[offsetA + i*n + k] += grad_value;
                        }
                    }
                }

                // 5. Call the previous backward if it exists (chaining backprop)
                if (this_backward_fn) {
                    this_backward_fn();
                }
            }

            if (other_requires_grad && other_grad) {

                // 1) Compute how many batch slices we have
                //    For example, if this_shape = [2, 3, 4, 5], then the last two dims 
                //    (4, 5) are (m, n), so the batch shape is [2, 3].
                //    The total number of slices = 2 * 3 = 6.
                size_t batch_count = 1;
                for (size_t i = 0; i + 2 < other_shape.size(); i++) {
                    batch_count *= other_shape[i];
                }

                // 2) Extract (n, p) from the last two dimensions of B
                //    In a typical matmul scenario:
                //      B has shape [..., n, p]
                const size_t n = other_shape[other_shape.size() - 2];
                const size_t p = other_shape[other_shape.size() - 1];

                // 3) For the same reason, A has shape [..., m, n],
                //    so we read m, n from A as well:
                const size_t m = this_shape[this_shape.size() - 2];
                // (We could re-check that the "n" dimension matches, etc.)

                // 4) Loop over each batch slice
                for (size_t b = 0; b < batch_count; b++) {
                    // Compute offsets in flattened data for A, B, C
                    // (assuming A, B, and C are fully batched with batch_count slices)
                    const size_t offsetA = b * (m * n);
                    const size_t offsetB = b * (n * p);
                    const size_t offsetC = b * (m * p);

                    // 4a) Accumulate gradient w.r.t. B in the b-th slice
                    //     dB[k, j] += sum_{i} [ A[i, k] * dC[i, j] ]
                    for (size_t k = 0; k < n; k++) {
                        for (size_t j = 0; j < p; j++) {
                            float grad_value = 0.0f;
                            for (size_t i = 0; i < m; i++) {
                                grad_value += A_data[offsetA + i*n + k]
                                            * result_grad->data[offsetC + i*p + j];
                            }
                            other_grad->data[offsetB + k*p + j] += grad_value;
                        }
                    }
                }

                // 5) Call the previous backward if it exists (chaining backprop)
                if (other_backward_fn) {
                    other_backward_fn();
                }
            }


        };

    }
    return result;
}

Tensor Tensor::sum(size_t dim) const {
    // 1. Check for valid dimension
    if (dim >= shape.size()) {
        throw std::invalid_argument("Invalid dimension for sum operation");
    }

    // 2. Compute new shape after reduction along `dim`
    //    e.g. if shape=[2,2,2,2], removing dim=0 => new_shape=[2,2,2]
    std::vector<size_t> new_shape = shape;
    new_shape.erase(new_shape.begin() + dim);

    // 3. Number of elements in the result
    size_t result_size = this->numel(new_shape);
    std::vector<float> result_data(result_size, 0.0f);

    //
    // 4. Forward pass: sum over dimension `dim` in the original tensor
    //
    //    For each index in [0..result_size), we interpret it as 
    //    a coordinate in the reduced shape. Then for each possible 
    //    index j in the dimension `dim`, we add up the original data.
    //
    for (size_t out_idx = 0; out_idx < result_size; ++out_idx) {
        // coords_reduced is a coordinate in the *new* shape (which excludes `dim`)
        std::vector<size_t> coords_reduced = unravel_index(out_idx, new_shape);

        // We'll create coords_full, which has one extra dimension inserted at `dim`
        // For example, if new_shape=[2,2,2] and dim=0 was removed from a 4D shape,
        // then coords_full will be [?,  coords_reduced[0], coords_reduced[1], coords_reduced[2]]
        // with ? in [0..shape[dim]-1].
        // We'll fill coords_full with the same values from coords_reduced, but with
        // a slot for the dimension `dim`.
        std::vector<size_t> coords_full(shape.size());
        
        // We'll copy the reduced coords into coords_full, skipping an index for `dim`
        // Something like:
        //   for i in [0..dim-1]: coords_full[i] = coords_reduced[i]
        //   coords_full[dim] = ...
        //   for i in [dim+1..end]: coords_full[i] = coords_reduced[i-1]
        size_t r_i = 0; // index for coords_reduced
        for (size_t full_i = 0; full_i < shape.size(); full_i++) {
            if (full_i == dim) {
                // We'll fill this inside the loop over j below
                continue;
            }
            coords_full[full_i] = coords_reduced[r_i];
            r_i++;
        }

        // Now sum over j in [0..shape[dim]).
        float sum_val = 0.0f;
        for (size_t j = 0; j < shape[dim]; j++) {
            coords_full[dim] = j;  // set the dimension we are summing over
            size_t orig_offset = ravel_index(coords_full, shape);
            sum_val += data[orig_offset];
        }

        result_data[out_idx] = sum_val;
    }

    //
    // 5. Create the reduced tensor
    //
    Tensor result(result_data, new_shape, requires_grad);

    //
    // 6. Set up backward function
    //
    if (result.requires_grad) {
        auto this_requires_grad = this->requires_grad;
        auto this_grad = this->grad;
        auto this_shape = this->shape;    // shape BEFORE reduction
        auto result_grad = result.grad;
        auto result_shape = result.shape; // shape AFTER reduction
        auto this_backward_fn = this->backward_fn;

        // We need to remember which dimension we reduced
        // so we can expand gradients back in backward pass
        auto reduced_dim = dim;

        result.backward_fn = [=]() mutable {
            if (this_requires_grad && this_grad) {
                //
                // We'll expand `result_grad` (shape = result_shape)
                // back to `this_shape`. 
                // For each element in the result_grad, we broadcast
                // that gradient across `this_shape[reduced_dim]` entries
                // in the original `this_grad`.
                //
                std::vector<float> expanded_grad(this_grad->data.size(), 0.0f);

                // for each index out_idx in [0..result_grad->data.size()),
                // unravel it in the *reduced* shape, 
                // then for j in [0..this_shape[reduced_dim]),
                // ravel back to an offset in the original shape
                for (size_t out_idx = 0; out_idx < result_grad->data.size(); ++out_idx) {
                    float grad_val = result_grad->data[out_idx];

                    // coords_reduced in the *result_shape*
                    auto coords_reduced = unravel_index(out_idx, result_shape);

                    // Expand to full coords
                    std::vector<size_t> coords_full(this_shape.size());
                    
                    // Fill coords_full except for the reduced_dim
                    {
                        size_t r_i = 0;
                        for (size_t full_i = 0; full_i < this_shape.size(); full_i++) {
                            if (full_i == reduced_dim) {
                                continue;
                            }
                            coords_full[full_i] = coords_reduced[r_i];
                            r_i++;
                        }
                    }

                    // Broadcast over the reduced dimension
                    for (size_t j = 0; j < this_shape[reduced_dim]; j++) {
                        coords_full[reduced_dim] = j;
                        size_t orig_offset = ravel_index(coords_full, this_shape);
                        expanded_grad[orig_offset] += grad_val;
                    }
                }

                // Now accumulate expanded_grad into this_grad->data
                for (size_t i = 0; i < expanded_grad.size(); ++i) {
                    this_grad->data[i] += expanded_grad[i];
                }
            }

            // Chain backward if there's a previous op
            if (this_backward_fn) {
                this_backward_fn();
            }
        };
    }

    return result;
}

Tensor Tensor::sum() const {
    return sum(shape.size() - 1);
}

Tensor Tensor::mean(size_t dim) const {
    // 1. Check for valid dimension
    if (dim >= shape.size()) {
        throw std::invalid_argument("Invalid dimension for mean operation");
    }

    // 2. Compute new shape after reduction along `dim`
    //    e.g. if shape=[2,2,3], removing dim=1 => new_shape=[2,3]
    std::vector<size_t> new_shape = shape;
    new_shape.erase(new_shape.begin() + dim);

    // 3. Number of elements in the reduced result
    size_t result_size = this->numel(new_shape);
    std::vector<float> result_data(result_size, 0.0f);

    // We’ll divide by this factor after summation
    float divisor = static_cast<float>(shape[dim]);

    //
    // 4. Forward pass: compute the mean by summing + dividing
    //
    //    For each index `out_idx` in [0..result_size),
    //    we interpret it as a coordinate in the reduced shape, 
    //    then for each j in [0..shape[dim]), we gather from the original 
    //    data and sum, then divide by shape[dim].
    //
    for (size_t out_idx = 0; out_idx < result_size; ++out_idx) {
        // coords_reduced is a coordinate in new_shape
        std::vector<size_t> coords_reduced = unravel_index(out_idx, new_shape);

        // We'll build coords_full for the original shape, 
        // but with one extra slot for the dimension `dim`.
        std::vector<size_t> coords_full(shape.size());

        // Fill coords_full except for the reduced_dim
        {
            size_t r_i = 0; 
            for (size_t full_i = 0; full_i < shape.size(); full_i++) {
                if (full_i == dim) {
                    // we'll fill in j later
                    continue;
                }
                coords_full[full_i] = coords_reduced[r_i];
                r_i++;
            }
        }

        float sum_val = 0.0f;
        // Sum over j in [0..shape[dim])
        for (size_t j = 0; j < shape[dim]; j++) {
            coords_full[dim] = j;
            size_t orig_offset = ravel_index(coords_full, shape);
            sum_val += data[orig_offset];
        }

        // Take the mean
        result_data[out_idx] = sum_val / divisor;
    }

    //
    // 5. Create the reduced (mean) tensor
    //
    Tensor result(result_data, new_shape, requires_grad);

    //
    // 6. If requires_grad, define the backward function
    //
    if (result.requires_grad) {
        // We'll capture everything needed in the lambda
        auto this_requires_grad = this->requires_grad;
        auto this_grad = this->grad;     // gradient buffer for the original
        auto this_shape = this->shape;   // original shape
        auto this_data = this->data;     // optional, if needed
        auto result_grad = result.grad;  // gradient buffer for the reduced tensor
        auto result_shape = result.shape;
        auto this_backward_fn = this->backward_fn;
        size_t reduced_dim = dim;        // dimension we reduced
        float divisor_f = divisor;       // shape[dim], as float

        result.backward_fn = [=]() mutable {
            if (this_requires_grad && this_grad) {
                //
                // We need to broadcast each element of result_grad
                // back across the `reduced_dim`, *but also divide*
                // was already done in forward, so in backward we *multiply*
                // by 1/divisor. Actually, let's see:
                //
                //   Mean is sum(...) / divisor
                //   d/dA of mean = 1/divisor for each element 
                //
                // So we do the normal broadcast, but each broadcasted grad
                // is scaled by (1/divisor).
                //
                std::vector<float> expanded_grad(this_grad->data.size(), 0.0f);

                // For each index in result_grad
                for (size_t out_idx = 0; out_idx < result_grad->data.size(); ++out_idx) {
                    float grad_val = result_grad->data[out_idx];
                    // coords in the reduced shape
                    auto coords_reduced = unravel_index(out_idx, result_shape);

                    // build full coords
                    std::vector<size_t> coords_full(this_shape.size());
                    {
                        size_t r_i = 0; 
                        for (size_t full_i = 0; full_i < this_shape.size(); full_i++) {
                            if (full_i == reduced_dim) {
                                continue;
                            }
                            coords_full[full_i] = coords_reduced[r_i];
                            r_i++;
                        }
                    }

                    // Broadcast over the dimension we reduced
                    for (size_t j = 0; j < this_shape[reduced_dim]; j++) {
                        coords_full[reduced_dim] = j;
                        size_t orig_offset = ravel_index(coords_full, this_shape);
                        // The partial derivative of mean wrt each element in that dim
                        // is 1/divisor. So multiply by grad_val * (1/divisor).
                        expanded_grad[orig_offset] += grad_val / divisor_f;
                    }
                }

                // Accumulate
                for (size_t i = 0; i < expanded_grad.size(); ++i) {
                    this_grad->data[i] += expanded_grad[i];
                }
            }

            // Chain backward if there was a previous operation
            if (this_backward_fn) {
                this_backward_fn();
            }
        };
    }

    return result;
}

Tensor Tensor::mean() const {
    return mean(shape.size() - 1);
}