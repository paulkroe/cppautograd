#include "cppgrad.h"

/* 
 * backward function for scalar tensors:
 * 1. Check if the tensor is scalar
 * 2. Check if the tensor requires gradient
 * 3. Initialize/Set the gradient to 1.0
 * 4. Call the backward function if tensor has a backward function
 */

void Tensor::backward() {
    if (this->numel(shape) != 1) {
        throw std::runtime_error("Backward only supported for scalar outputs");
    }
    if (!requires_grad) {
        throw std::runtime_error("This tensor does not require gradient");
    }

    // Initialize gradient as 1.0 if missing
    if (!grad) {
        grad = std::make_shared<Tensor>(std::vector<float>(data.size(), 1.0f), shape, false);
    } else {
        std::fill(grad->data.begin(), grad->data.end(), 1.0f);
    }

    // Step 1: Use raw pointers for tracking visited tensors
    std::unordered_set<Tensor*> visited;  // Now stores raw pointers
    std::unordered_set<int> visited_id;
    std::stack<Tensor*> stack;
    stack.push(this);

    while (!stack.empty()) {
        Tensor* current = stack.top();
        stack.pop();

        // Skip processing if already visited
        if (visited.count(current)) continue;
        if (visited_id.count(current->id)) continue;
        visited.insert(current);  // Track the raw pointer
        visited_id.insert(current->id);

        // Ensure it has gradient storage
        if (!current->grad) {
            current->grad = std::make_shared<Tensor>(
                std::vector<float>(current->data.size(), 0.0f), current->shape, false
            );
        }

        // Execute the backward function if it exists
        if (current->backward_fn) {
            current->backward_fn();
        }

        // Push parent nodes onto the stack, using raw pointers
        for (const auto& parent : current->parents) {
            stack.push(parent.get());  // Get the raw pointer from shared_ptr
        }
    }
}

std::vector<size_t> Tensor::unflatten_index(size_t flat_index, const std::vector<size_t>& shape) {
    std::vector<size_t> multi_index(shape.size(), 0);
    size_t temp = flat_index;

    for (int i = shape.size() - 1; i >= 0; --i) {
        multi_index[i] = temp % shape[i];
        temp /= shape[i];
    }

    return multi_index;
}

std::vector<float> Tensor::reduce_grad(const std::vector<float>& grad, 
                               const std::vector<size_t>& grad_shape, 
                               const std::vector<size_t>& original_shape) {
    std::vector<float> reduced_grad(Tensor::numel(original_shape), 0.0f);

    for (size_t i = 0; i < grad.size(); ++i) {
        std::vector<size_t> multi_index = Tensor::unflatten_index(i, grad_shape);
        size_t reduced_index = map_index(multi_index, original_shape);
        reduced_grad[reduced_index] += grad[i];
    }

    return reduced_grad;
}

/*
 * Matrix multiplication of two tensors, with broadcasting.
 * We assume 
 * A has shape [..., m, n]
 * B has shape [..., x, p]
 * such that n == x
 * thus:
 * C = A.matul(B) has shape [..., m, p]
 */
Tensor Tensor::matmul(const Tensor &other) const {
    // 1. Validate that both have at least 2 dims
    if (shape.size() < 2 || other.shape.size() < 2) {
        throw std::invalid_argument(
            "Both tensors must have at least 2 dimensions for matmul."
        );
    }

    // 2. If both are strictly 2D, do a simpler 2D matmul (no broadcasting)
    if (shape.size() == 2 && other.shape.size() == 2) {
        // 2D matrix multiply: A in [m, n], B in [n, p]
        size_t m = shape[0];
        size_t n = shape[1];
        size_t x = other.shape[0];
        size_t p = other.shape[1];

        if (n != x) {
            throw std::invalid_argument(
                "Inner dimensions do not match for 2D matmul."
            );
        }

        // Construct output shape [m, p]
        std::vector<size_t> result_shape = {m, p};
        std::vector<float> result_data(m * p, 0.0f);

        // Forward pass
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < p; j++) {
                float sum = 0.0f;
                for (size_t k = 0; k < n; k++) {
                    sum += data[i*n + k] * other.data[k*p + j];
                }
                result_data[i*p + j] = sum;
            }
        }

        // Create output tensor
        std::shared_ptr<Tensor> result = std::make_shared<Tensor>(result_data, result_shape, requires_grad || other.requires_grad);

        // Backward pass
        if (result->requires_grad) {
            // Capture everything needed by value
            auto A_grad          = this->grad;
            auto B_grad          = other.grad;
            auto result_grad     = result->grad;
            auto A_data          = data;
            auto B_data          = other.data;
            auto this_backward_fn  = this->backward_fn;
            auto other_backward_fn = other.backward_fn;

            bool A_req_grad = requires_grad;     // local copy
            bool B_req_grad = other.requires_grad;

            if (A_req_grad)
                result->parents.push_back(std::make_shared<Tensor>(*this));
            if (B_req_grad)
                result->parents.push_back(std::make_shared<Tensor>(other));

            result->backward_fn = [=]() mutable {
                // dA = dC * B^T
                if (A_req_grad && A_grad) {
                    for (size_t i = 0; i < m; i++) {
                        for (size_t k = 0; k < n; k++) {
                            float grad_val = 0.0f;
                            for (size_t j = 0; j < p; j++) {
                                grad_val += result_grad->data[i*p + j]
                                          * B_data[k*p + j];
                            }
                            A_grad->data[i*n + k] += grad_val;
                        }
                    }
                }

                // dB = A^T * dC
                if (B_req_grad && B_grad) {
                    for (size_t k = 0; k < n; k++) {
                        for (size_t j = 0; j < p; j++) {
                            float grad_val = 0.0f;
                            for (size_t i = 0; i < m; i++) {
                                grad_val += A_data[i*n + k]
                                          * result_grad->data[i*p + j];
                            }
                            B_grad->data[k*p + j] += grad_val;
                        }
                    }
                }
            };
        }

        return *result;
    }
    else {
        // batched case
        size_t m = shape[shape.size() - 2];
        size_t n = shape[shape.size() - 1];
        size_t x = other.shape[other.shape.size() - 2];
        size_t p = other.shape[other.shape.size() - 1];

        if (n != x) {
            throw std::invalid_argument(
                "Inner dimensions of the tensors must match for matmul."
            );
        }

        // (A) Determine batch shape
        std::vector<size_t> this_batch_shape;
        std::vector<size_t> other_batch_shape;
        std::vector<size_t> batch_shape;

        // If either tensor has extra dims (beyond the last 2),
        // we must handle broadcasting those leading dims
        if (shape.size() > 2 || other.shape.size() > 2) {
            this_batch_shape =
                std::vector<size_t>(shape.begin(), shape.end() - 2);
            other_batch_shape =
                std::vector<size_t>(other.shape.begin(), other.shape.end() - 2);

            // broadcast(...) is assumed to return the broadcasted shape
            batch_shape = broadcast(this_batch_shape, other_batch_shape);
            if (batch_shape.empty()) {
                // Not broadcastable
                this->print_shape();
                other.print_shape();
                throw std::invalid_argument(
                    "Tensors not broadcastable for matmul."
                );
            }
        }
       
        // else: no batch dims => batch_shape stays empty => batch_size = 1

        // (B) Construct result shape: batch_shape + [m, p]
        auto result_shape = batch_shape;
        result_shape.push_back(m);
        result_shape.push_back(p);

        // (C) Forward pass
        size_t batch_size = 1;
        for (auto d : batch_shape) {
            batch_size *= d;
        }

        size_t total_elems = batch_size * m * p;
        std::vector<float> result_data(total_elems, 0.0f);

        // Multiply each batch
        for (size_t b = 0; b < batch_size; ++b) {
            // Convert b to multi-index in the broadcast shape
            std::vector<size_t> batch_index = Tensor::unflatten_index(b, batch_shape);

            // Figure out which offsets inside A and B this corresponds to
            size_t A_offset = map_index(batch_index, this_batch_shape) * m * n;
            size_t B_offset = map_index(batch_index, other_batch_shape) * n * p;
            size_t C_offset = b * (m * p);

            // Standard matrix multiply for each batch
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

        // (D) Build result
        std::shared_ptr<Tensor> result = std::make_shared<Tensor>(result_data, result_shape, requires_grad || other.requires_grad);

        // (E) Backward pass
        if (result->requires_grad) {
            // Capture everything needed
            auto this_requires_grad  = requires_grad;
            auto other_requires_grad = other.requires_grad;

            auto this_grad          = this->grad;
            auto other_grad         = other.grad;
            auto result_grad        = result->grad;

            auto A_data             = data;
            auto B_data             = other.data;

            auto this_backward_fn   = this->backward_fn;
            auto other_backward_fn  = other.backward_fn;

            // Save the original shapes for correct “reduce”
            auto saved_this_shape  = shape;   // e.g. [batch_dims..., m, n]
            auto saved_other_shape = other.shape;
            auto saved_batch_shape = batch_shape;

            // We also store m,n,p because they’re needed in the lambda
            size_t mm = m;
            size_t nn = n;
            size_t pp = p;

            if (this_requires_grad)
                result->parents.push_back(std::make_shared<Tensor>(*this));
            
            if (other_requires_grad)
                result->parents.push_back(std::make_shared<Tensor>(other));

            result->backward_fn = [=]() mutable {
                // (1) We compute how many total “batches” we did
                size_t batch_size_local = 1;
                for (auto d : saved_batch_shape) {
                    batch_size_local *= d;
                }

                // (2) dA = dC * B^T (batched)
                if (this_requires_grad && this_grad) {
                    // Accumulate into this_grad->data, shape is effectively
                    // broadcasted: batch_shape + [m, n]
                    for (size_t b = 0; b < batch_size_local; b++) {
                        std::vector<size_t> batch_idx =
                            Tensor::unflatten_index(b, saved_batch_shape);

                        size_t A_offset = map_index(batch_idx,
                            std::vector<size_t>(saved_this_shape.begin(),
                                                saved_this_shape.end()-2))
                                          * (mm * nn);

                        size_t B_offset = map_index(batch_idx,
                            std::vector<size_t>(saved_other_shape.begin(),
                                                saved_other_shape.end()-2))
                                          * (nn * pp);

                        size_t C_offset = b * (mm * pp);

                        for (size_t i = 0; i < mm; i++) {
                            for (size_t k = 0; k < nn; k++) {
                                float grad_value = 0.0f;
                                for (size_t j = 0; j < pp; j++) {
                                    grad_value += result_grad->data[C_offset + i*pp + j]
                                                * B_data[B_offset + k*pp + j];
                                }
                                this_grad->data[A_offset + i*nn + k] += grad_value;
                            }
                        }
                    }

                    // Now reduce from shape = broadcasted (batch_shape + [m, n])
                    // down to the original this->shape if it had broadcasted dims
                    // Build the “from_shape” for the gradient
                    std::vector<size_t> from_shape = saved_batch_shape; // the broadcast batch
                    from_shape.push_back(mm);
                    from_shape.push_back(nn);

                    // e.g. from_shape might be [2,4,5, m,n] if broadcast
                    // but saved_this_shape is the real [1,4,5, m,n], etc.
                    this_grad->data = reduce_grad(
                        this_grad->data,
                        from_shape,
                        saved_this_shape
                    );

                }

                // (3) dB = A^T * dC (batched)
                if (other_requires_grad && other_grad) {
                    for (size_t b = 0; b < batch_size_local; b++) {
                        std::vector<size_t> batch_idx =
                            Tensor::unflatten_index(b, saved_batch_shape);

                        size_t A_offset = map_index(batch_idx,
                            std::vector<size_t>(saved_this_shape.begin(),
                                                saved_this_shape.end()-2))
                                          * (mm * nn);

                        size_t B_offset = map_index(batch_idx,
                            std::vector<size_t>(saved_other_shape.begin(),
                                                saved_other_shape.end()-2))
                                          * (nn * pp);

                        size_t C_offset = b * (mm * pp);

                        for (size_t k = 0; k < nn; k++) {
                            for (size_t j = 0; j < pp; j++) {
                                float grad_value = 0.0f;
                                for (size_t i = 0; i < mm; i++) {
                                    grad_value += A_data[A_offset + i*nn + k]
                                                * result_grad->data[C_offset + i*pp + j];
                                }
                                other_grad->data[B_offset + k*pp + j] += grad_value;
                            }
                        }
                    }
                    // reduce from broadcast shape = batch_shape + [n, p]
                    // down to the original other.shape
                    std::vector<size_t> from_shape = saved_batch_shape;
                    from_shape.push_back(nn);
                    from_shape.push_back(pp);

                    other_grad->data = reduce_grad(
                        other_grad->data,
                        from_shape,
                        saved_other_shape
                    );

                }
            };
        }

        // Return the final result
        return *result;
    }
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
    std::shared_ptr<Tensor> result = std::make_shared<Tensor>(result_data, new_shape, requires_grad);
    //
    // 6. Set up backward function
    //
    if (result->requires_grad) {
        auto this_requires_grad = this->requires_grad;
        auto this_grad = this->grad;
        auto this_shape = this->shape;    // shape BEFORE reduction
        auto result_grad = result->grad;
        auto result_shape = result->shape; // shape AFTER reduction
        auto this_backward_fn = this->backward_fn;

        // We need to remember which dimension we reduced
        // so we can expand gradients back in backward pass
        auto reduced_dim = dim;

        if (this_requires_grad)
            result->parents.push_back(std::make_shared<Tensor>(*this));

        result->backward_fn = [=]() mutable {
            if (this_requires_grad && this_grad) {
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
        };
    }

    return *result;
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
    std::shared_ptr<Tensor> result = std::make_shared<Tensor>(result_data, new_shape, requires_grad);

    //
    // 6. If requires_grad, define the backward function
    //
    if (result->requires_grad) {
        // We'll capture everything needed in the lambda
        auto this_requires_grad = this->requires_grad;
        auto this_grad = this->grad;     // gradient buffer for the original
        auto this_shape = this->shape;   // original shape
        auto this_data = this->data;     // optional, if needed
        auto result_grad = result->grad;  // gradient buffer for the reduced tensor
        auto result_shape = result->shape;
        auto this_backward_fn = this->backward_fn;
        size_t reduced_dim = dim;        // dimension we reduced
        float divisor_f = divisor;       // shape[dim], as float

        if (this_requires_grad)
            result->parents.push_back(std::make_shared<Tensor>(*this));

        result->backward_fn = [=]() mutable {
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

        };
    }

    return *result;
}

Tensor Tensor::mean() const {
    return mean(shape.size() - 1);
}

Tensor Tensor::exp() const{
    // 1) Compute exp elementwise
    std::vector<float> forward_data(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        forward_data[i] = std::exp(data[i]);
    }

    // 2) Create the output tensor
    std::shared_ptr<Tensor> result = std::make_shared<Tensor>(forward_data, shape, requires_grad);
    
    // 3) If we need gradients, set up the backward function
    if (result->requires_grad) {
        // Capture only what's needed in the lambda
        auto this_requires_grad = this->requires_grad;
        auto this_grad    = this->grad;   // Gradient buffer of the input
        auto this_data    = this->data;   // For completeness, if needed
        auto out_data     = result->data;  // The exp(...) values we just computed
        auto result_grad  = result->grad;
        auto this_backward_fn = this->backward_fn;

        if (this_requires_grad)
            result->parents.push_back(std::make_shared<Tensor>(*this));


        result->backward_fn = [=]() mutable {
            if (this_requires_grad && this_grad) {
                for (size_t i = 0; i < this_data.size(); i++) {
                    // out_data[i] is exp(this_data[i])
                    this_grad->data[i] += result_grad->data[i] * out_data[i];
                }
            }
        };
    }

    return *result;
}

Tensor Tensor::log() const {
    // 1) Compute exp elementwise
    std::vector<float> forward_data(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        forward_data[i] = std::log(data[i]);
    }

    // 2) Create the output tensor
    std::shared_ptr<Tensor> result = std::make_shared<Tensor>(forward_data, shape, requires_grad);

    // 3) If we need gradients, set up the backward function
    if (result->requires_grad) {
        // Capture only what's needed in the lambda
        auto this_requires_grad = this->requires_grad;
        auto this_grad    = this->grad;   // Gradient buffer of the input
        auto this_data    = this->data;   // For completeness, if needed
        auto out_data     = result->data;  // The log(...) values we just computed
        auto result_grad  = result->grad;
        auto this_backward_fn = this->backward_fn;

        if (this_requires_grad)
            result->parents.push_back(std::make_shared<Tensor>(*this));

        result->backward_fn = [=]() mutable {
            if (this_requires_grad && this_grad) {
                for (size_t i = 0; i < this_data.size(); i++) {
                    this_grad->data[i] += result_grad->data[i] / this_data[i];
                }
            }
        };
    }

    return *result;
}


Tensor Tensor::softmax(size_t dim) const {
    Tensor exp_tensor = this->exp(); 
    Tensor sum_exp = exp_tensor.sum(dim) + Tensor({1e-6}, {1}); // Add a small epsilon to avoid division by zero
    sum_exp.shape.push_back(1); // Add a new dimension for broadcasting 

    return exp_tensor / sum_exp;
}

Tensor Tensor::softmax() const {
    return softmax(shape.size() - 1);
}

Tensor Tensor::onehot_encode(size_t num_classes) const {
    size_t result_size = this->numel(shape) * num_classes;
    std::vector<float> result_data(result_size, 0.0f);
    
    std::vector<size_t> result_shape = shape;
    result_shape.push_back(num_classes);
    
    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i] >= num_classes) {
            throw std::invalid_argument("Value out of bounds for one-hot encoding");
        }
        std::vector<size_t> multi_index(shape.size(), 0);
        size_t temp = i;
        for (int j = shape.size() - 1; j >= 0; --j) {
            multi_index[j] = temp % shape[j];
            temp /= shape[j];
        }
        multi_index.push_back(data[i]);
        size_t index = map_index(multi_index, result_shape);

        result_data[index] = 1.0f;
    }

    std::shared_ptr<Tensor> result = std::make_shared<Tensor>(result_data, result_shape, requires_grad);

    if (result->requires_grad) {
        auto this_requires_grad = this->requires_grad;
        auto this_grad = this->grad;
        auto result_grad = result->grad;
        auto this_data = this->data;
        auto this_shape = this->shape;
        auto this_backward_fn = this->backward_fn;

        if (this_requires_grad)
            result->parents.push_back(std::make_shared<Tensor>(*this));

        result->backward_fn = [=]() mutable {
            if (this_requires_grad && this_grad) {
                for (size_t i = 0; i < this_data.size(); ++i) {
                    std::vector<size_t> multi_index(shape.size(), 0);
                    size_t temp = i;
                    for (int j = shape.size() - 1; j >= 0; --j) {
                        multi_index[j] = temp % shape[j];
                        temp /= shape[j];
                    }
                    for (size_t j = 0; j < num_classes; ++j) {
                        multi_index.push_back(j);
                        size_t index = map_index(multi_index, result_shape);
                        this_grad->data[i] += result_grad->data[index];
                        multi_index.pop_back();
                    }
                }
            }
        };
    }

    return *result; 

}

Tensor Tensor::relu() const{
    // 1) Compute exp elementwise
    std::vector<float> forward_data(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        forward_data[i] = data[i] > 0 ? data[i] : 0;
    }

    // 2) Create the output tensor
    std::shared_ptr<Tensor> result = std::make_shared<Tensor>(forward_data, shape, requires_grad);

    // 3) If we need gradients, set up the backward function
    if (result->requires_grad) {
        // Capture only what's needed in the lambda
        auto this_requires_grad = this->requires_grad;
        auto this_grad    = this->grad;   // Gradient buffer of the input
        auto this_data    = this->data;   // For completeness, if needed
        auto out_data     = result->data; 
        auto result_grad  = result->grad;
        auto this_backward_fn = this->backward_fn;

        if (this_requires_grad)
            result->parents.push_back(std::make_shared<Tensor>(*this));

        result->backward_fn = [=]() mutable {
            if (this_requires_grad && this_grad) {
                for (size_t i = 0; i < this_data.size(); i++) {
                    this_grad->data[i] += result_grad->data[i] * (this_data[i] > 0 ? 1 : 0);
                }
            }
        };
    }

    return *result;
}

Tensor CrossEntropyLoss(const Tensor& y_pred, const Tensor& y_true) {
    // 1) Compute the softmax of y_pred along the last dimension (class axis)
    Tensor y_pred_softmax = y_pred.softmax(y_pred.shape.size() - 1);  

    // 2) Convert true labels into one-hot representation
    Tensor y_true_one_hot = y_true.onehot_encode(y_pred.shape.back());

    // 3) Compute negative log-likelihood: - sum(one_hot * log(softmax)) along class axis
    Tensor neg_log_likelihood = -(y_true_one_hot * y_pred_softmax.log()).sum(y_pred.shape.size() - 1);

    // 4) Compute mean loss over batch dimension
    Tensor loss = neg_log_likelihood.mean();

    return loss;
}