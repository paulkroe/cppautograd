#include "cppgrad.h"
#include <iostream>

/* 
 * backward function for scalar tensors:
 * 1. Check if the tensor is scalar
 * 2. Check if the tensor requires gradient
 * 3. Initialize/Set the gradient to 1.0
 * 4. Call the backward function if tensor has a backward function
 */
void Tensor::backward() {
    /* Check if the tensor is a scalar */
    if (this->numel(shape) != 1) {
        throw std::runtime_error("Backward only supported for scalar outputs");
    }

    /* Check if the tensor requires gradient */
    if (!requires_grad) {
        throw std::runtime_error("This tensor does not require gradient");
    }
    /* Initialize gradient if necessary */
    if (!grad) {
        grad = std::make_shared<Tensor>(std::vector<float>(data.size(), 1.0f), shape, false);
    } else {
        /* Ensure the gradient is explicitly set to 1.0 for the target */
        std::fill(grad->data.begin(), grad->data.end(), 1.0f);
    }

    /* Call the backward function of the target tensor */
    if (backward_fn) {
        backward_fn();
    }
}

/* 
 * Helper function to infer broadcast shape 
 * To shapes are broadcastable 
 * When iterating over the dimension sizes, starting at the trailing dimension,
 * the dimension sizes must either be equal, one of them is 1, or one of them does not exist
 */
std::vector<size_t> broadcast(const std::vector<size_t>& shape1, const std::vector<size_t>& shape2) {
    size_t max_dims = std::max(shape1.size(), shape2.size());
    std::vector<size_t> result(max_dims, 1);

    for (size_t i = 0; i < max_dims; ++i) {
        size_t dim1 = i < shape1.size() ? shape1[shape1.size() - 1 - i] : 1;
        size_t dim2 = i < shape2.size() ? shape2[shape2.size() - 1 - i] : 1;

        if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
            return {};
        }

        result[max_dims - 1 - i] = std::max(dim1, dim2);
    }

    return result;
}

/*
 * Helper function to print tensor shape
 */
void printShapes(const std::vector<size_t>& shape1, const std::vector<size_t>& shape2) {
    std::cout << "this->shape ";
    for (auto s : shape1) {
        std::cout << s << " ";
    }
    std::cout << " vs. other.shape ";
    for (auto s : shape2) {
        std::cout << s << " ";
    }
    std::cout << std::endl;
}

/* 
 * Overloaded + operator, does not support broadcasting
 * 1. Check if the shapes of the tensors match or the tensors are broadcastable
 * 2. Perform elementwise addition
 * 3. Construct the result tensor
 * 4. If necessary, set up the backward function
 */
Tensor Tensor::operator+(const Tensor& other) const{
    /* Check if shapes match */
    if (shapes_equal(this->shape, other.shape)) {
        /* Perform addition */
        std::vector<float> result_data(data.size());
        for (size_t i = 0; i < data.size(); i++) {
            result_data[i] = data[i] + other.data[i];
        }

        /* Construct result tensor */
        Tensor result(result_data, shape, requires_grad || other.requires_grad);

        /* Construct backward function */
        if (result.requires_grad) {
            /* 
            * Capture references to required tensor data and metadata
            * (e.g., grad pointers, backward functions) to ensure they remain
            * accessible during the backward pass, even after the forward
            * computation has completed.
            */
            auto sz = data.size();
            auto this_requires_grad = this->requires_grad;
            auto other_requires_grad = other.requires_grad;
            auto this_grad = this->grad;
            auto other_grad = other.grad;
            auto this_backward_fn = this->backward_fn;
            auto other_backward_fn = other.backward_fn;
            auto result_grad = result.grad;

            result.backward_fn = [sz, 
                                this_requires_grad, other_requires_grad,
                                this_grad, other_grad,
                                this_backward_fn, other_backward_fn,
                                result_grad]() {
                
                /* 
                * If this tensor requires its gradient and
                * the gradient vector exists, update its gradient 
                */
                if (this_requires_grad && this_grad) {
                    for (size_t i = 0; i < sz; i++) {
                        this_grad->data[i] += result_grad->data[i];
                    }
                    /* Call the previous backward function if it exists */
                    if (this_backward_fn) this_backward_fn();
                }

                /* 
                * If the other tensor requires its gradient and
                * the gradient vector exists, update its gradient 
                */
                if (other_requires_grad && other_grad) {
                    for (size_t i = 0; i < sz; i++) {
                        other_grad->data[i] += result_grad->data[i];
                    }
                    /* Call the previous backward function if it exists */
                    if (other_backward_fn) other_backward_fn();
                }
            };
        }

        return result;       
    }
    else {
        std::vector<size_t> result_shape;
        result_shape = broadcast(this->shape, other.shape);
        
        if (result_shape.empty()) {
            printShapes(this->shape, other.shape);
            throw std::invalid_argument("Tensor shapes must be broadcastable for addition.");
        }
               
        /* Perform addition with broadcasting */
        size_t result_size = numel(result_shape);
        std::vector<float> result_data(result_size);

        for (size_t i = 0; i < result_size; ++i) {
            // Compute the multi-dimensional index in the result shape
            std::vector<size_t> multi_index(result_shape.size(), 0);
            size_t temp = i;
            for (int j = result_shape.size() - 1; j >= 0; --j) {
                multi_index[j] = temp % result_shape[j];
                temp /= result_shape[j];
            }

            // Map to indices in the original tensors
            size_t index_a = map_index(multi_index, this->shape);
            size_t index_b = map_index(multi_index, other.shape);

            // Perform the addition
            result_data[i] = this->data[index_a] + other.data[index_b];
        }

        Tensor result = Tensor(result_data, result_shape, this->requires_grad || other.requires_grad);

        if (result.requires_grad) {

            auto sz = data.size();
            auto this_requires_grad = this->requires_grad;
            auto other_requires_grad = other.requires_grad;
            auto this_grad = this->grad;
            auto other_grad = other.grad;
            auto this_backward_fn = this->backward_fn;
            auto other_backward_fn = other.backward_fn;
            auto result_grad = result.grad;

            result.backward_fn = [sz,
                                this_requires_grad, other_requires_grad,
                                this_grad, other_grad,
                                this_backward_fn, other_backward_fn,
                                result_grad, this_shape = this->shape, other_shape = other.shape,
                                result_shape = result.shape]() {

                // Lambda to compute the reduction for broadcasting
                auto reduce_broadcasted_grad = [](const std::vector<float>& grad_data,
                                  const std::vector<size_t>& grad_shape,
                                  const std::vector<size_t>& original_shape) -> std::vector<float>{
                    // Number of elements, not just dimension count
                    size_t original_numel = numel(original_shape);

                    // Allocate the full size
                    std::vector<float> reduced_grad(original_numel, 0.0f);

                    // Accumulate
                    for (size_t i = 0; i < grad_data.size(); ++i) {
                        std::vector<size_t> multi_index(grad_shape.size());
                        size_t temp = i;

                        // Convert the linear index `i` into a multi-dimensional index,
                        // by repeatedly dividing by each dimension size from right to left.
                        for (int j = grad_shape.size() - 1; j >= 0; --j) {
                            multi_index[j] = temp % grad_shape[j];
                            temp /= grad_shape[j];
                        }

                        // Convert multi_index to linear index in original_shape
                        size_t original_index = 0, stride = 1;
                        for (int j = original_shape.size() - 1; j >= 0; --j) {
                            size_t dim_index = multi_index[j] % original_shape[j];
                            original_index += dim_index * stride;
                            stride *= original_shape[j];
                        }

                        reduced_grad[original_index] += grad_data[i];
                    }

                    return reduced_grad;
                };


                // Backpropagate gradient for the first tensor
                if (this_requires_grad && this_grad) {
                    auto reduced_grad = reduce_broadcasted_grad(result_grad->data, result_shape, this_shape);
                    for (size_t i = 0; i < reduced_grad.size(); ++i) {
                        this_grad->data[i] += reduced_grad[i];
                    }
                    if (this_backward_fn) this_backward_fn();
                }

                // Backpropagate gradient for the second tensor
                if (other_requires_grad && other_grad) {
                    auto reduced_grad = reduce_broadcasted_grad(result_grad->data, result_shape, other_shape);
                    for (size_t i = 0; i < reduced_grad.size(); ++i) {
                        other_grad->data[i] += reduced_grad[i];
                    }
                    if (other_backward_fn) other_backward_fn();
                }
            };

        }

        return result;
    }

}

/* 
 * Overloaded unary - operator
 * 1. Negate the data
 * 2. Negate the gradient if necessary
 * 3. Construct the result tensor
 * 4. If necessary, set up the backward function
 */
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

    /* Construct result tensor */
    Tensor result(negated_data, shape, requires_grad, negated_grad);

    /* Construct backward function */
    if (result.requires_grad) {
         /* 
         * Capture references to required tensor data and metadata
         * (e.g., grad pointers, backward functions) to ensure they remain
         * accessible during the backward pass, even after the forward
         * computation has completed.
         */
        auto sz = data.size();
        auto this_requires_grad = this->requires_grad;
        auto this_grad = this->grad;
        auto this_backward_fn = this->backward_fn;
        auto result_grad = result.grad;

        result.backward_fn = [sz,
                              this_requires_grad, this_grad,
                              this_backward_fn, result_grad]() {
            if (this_requires_grad && this_grad) {
               /*
                * If this tensor requires its gradient and
                * the gradient vector exists, update its gradient 
                */
                for (size_t i = 0; i < sz; i++) {
                    this_grad->data[i] -= result_grad->data[i];
                }
                /* Call the previous backward function if it exists */
                if (this_backward_fn) this_backward_fn();
            }
        };
    }

    return result;
}

/* 
 * Overloaded binary - operator
 * using unary - and + operators
 * 1. Check if shapes match
 * 2. this - other = this + (-other)
 */
Tensor Tensor::operator-(const Tensor& other) const{
    /* check if shapes match */
    if (!shapes_equal(this->shape, other.shape)) {
        throw std::invalid_argument("Tensor shapes must match for subtraction.");
    }
    /* a - b =  a + (-b) */
    return *this + (-other);
}

/* TODO: implement support for broadcasting */
/* 
 * Overloaded binary * operator 
 * 1. Check if shapes match
 * 2. Perform elementwise multiplication
 */
Tensor Tensor::operator*(const Tensor& other) const{
    /* Check if shapes match */
    if (!shapes_equal(this->shape, other.shape)) {
        throw std::invalid_argument("Tensor shapes must match for multiplication.");
    }

    /* Perform elementwise multiplication */
    std::vector<float> result_data(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        result_data[i] = data[i] * other.data[i];
    }

    /* Construct result tensor */
    Tensor result(result_data, shape, requires_grad || other.requires_grad);
    
    /* Construct backward function */
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
           /*
            * If this tensor requires its gradient and
            * the gradient vector exists, update its gradient 
            */
            if (this_requires_grad && this_grad) {
                for (size_t i = 0; i < sz; i++) {
                    this_grad->data[i] += other_data[i] * result_grad->data[i];
                }
                /* Call the previous backward function if it exists */
                if (this_backward_fn) this_backward_fn();
            }
            /*
             * If other tensor requires its gradient and
             * the gradient vector exists, update its gradient 
             */
            if (other_requires_grad && other_grad) {
                for (size_t i = 0; i < sz; i++) {
                    other_grad->data[i] += this_data[i] * result_grad->data[i];
                }
                /* Call the previous backward function if it exists */
                if (other_backward_fn) other_backward_fn();
            }
        };
    }

    return result;
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
        size_t reduced_index = Tensor::map_index(multi_index, original_shape);
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
        throw std::invalid_argument("Both tensors must have at least 2 dimensions for matmul.");
    }

    // 2. If both are strictly 2D, do a simpler 2D matmul (no broadcasting)
    if (shape.size() == 2 && other.shape.size() == 2) {
        // 2D matrix multiply: A in [m, n], B in [n, p]
        size_t m = shape[0];
        size_t n = shape[1];
        size_t x = other.shape[0];
        size_t p = other.shape[1];

        if (n != x) {
            throw std::invalid_argument("Inner dimensions do not match for 2D matmul.");
        }

        // Construct output shape [m, p]
        std::vector<size_t> result_shape = {m, p};
        std::vector<float> result_data(m * p, 0.0f);

        // Forward pass: standard triple-nested loop
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
        bool out_requires_grad = (requires_grad || other.requires_grad);
        Tensor result(result_data, result_shape, out_requires_grad);

        // Backward pass
        if (out_requires_grad) {
            auto A_grad = this->grad;
            auto B_grad = other.grad;
            auto result_grad = result.grad;

            auto A_data = data;
            auto B_data = other.data;

            auto this_backward_fn = this->backward_fn;
            auto other_backward_fn = other.backward_fn;

            result.backward_fn = [=]() mutable {
                // dA = dC * B^T
                if (requires_grad && A_grad) {
                    for (size_t i = 0; i < m; i++) {
                        for (size_t k = 0; k < n; k++) {
                            float grad_val = 0.0f;
                            for (size_t j = 0; j < p; j++) {
                                grad_val += result_grad->data[i*p + j] * B_data[k*p + j];
                            }
                            A_grad->data[i*n + k] += grad_val;
                        }
                    }
                    // Chain back
                    if (this_backward_fn) this_backward_fn();
                }

                // dB = A^T * dC
                if (other.requires_grad && B_grad) {
                    for (size_t k = 0; k < n; k++) {
                        for (size_t j = 0; j < p; j++) {
                            float grad_val = 0.0f;
                            for (size_t i = 0; i < m; i++) {
                                grad_val += A_data[i*n + k] * result_grad->data[i*p + j];
                            }
                            B_grad->data[k*p + j] += grad_val;
                        }
                    }
                    // Chain back
                    if (other_backward_fn) other_backward_fn();
                }
            };
        }

        return result;
    }
    else {

        size_t m = shape[shape.size() - 2];
        size_t n = shape[shape.size() - 1];
        size_t x = other.shape[other.shape.size() - 2];
        size_t p = other.shape[other.shape.size() - 1];

        if (n != x) {
            throw std::invalid_argument("Inner dimensions of the tensors must match for matmul.");
        }

        // (A) Determine batch shape
        std::vector<size_t> this_batch_shape;
        std::vector<size_t> other_batch_shape;
        std::vector<size_t> batch_shape;

        if (shape.size() > 2 || other.shape.size() > 2) {
            this_batch_shape = std::vector<size_t>(shape.begin(), shape.begin() + (shape.size() - 2));
            other_batch_shape = std::vector<size_t>(other.shape.begin(), other.shape.begin() + (other.shape.size() - 2));

            batch_shape = broadcast(this_batch_shape, other_batch_shape);
            if (batch_shape.empty()) {
                this->print_shape();
                other.print_shape();
                throw std::invalid_argument("Tensors not broadcastable for matmul.");
            }
        }
        // else: no batch dims, batch_shape remains empty => batch_size = 1

        // (B) Construct result shape
        auto result_shape = batch_shape;
        result_shape.push_back(m);
        result_shape.push_back(p);

        // (C) Forward pass
        size_t batch_size = 1;
        for (auto d : batch_shape) batch_size *= d;

        size_t total_elems = batch_size * m * p;
        std::vector<float> result_data(total_elems, 0.0f);

        for (size_t b = 0; b < batch_size; ++b) {
            std::vector<size_t> batch_index = Tensor::unflatten_index(b, batch_shape);
            size_t A_offset = map_index(batch_index, this_batch_shape) * m * n;
            size_t B_offset = map_index(batch_index, other_batch_shape) * n * p;
            size_t C_offset = b * (m * p);

            for (size_t i = 0; i < m; i++) {
                for (size_t j = 0; j < p; j++) {
                    float sum = 0.0f;
                    for (size_t k = 0; k < n; k++) {
                        sum += data[A_offset + i*n + k] * other.data[B_offset + k*p + j];
                    }
                    result_data[C_offset + i*p + j] = sum;
                }
            }
        }

        // (D) Build result
        bool out_requires_grad = (requires_grad || other.requires_grad);
        Tensor result(result_data, result_shape, out_requires_grad);

        // (E) Backward pass
        if (out_requires_grad) {
            auto this_requires_grad = requires_grad;
            auto other_requires_grad = other.requires_grad;
            auto this_grad = this->grad;
            auto other_grad = other.grad;
            auto result_grad = result.grad;

            auto A_data = data;
            auto B_data = other.data;

            auto this_backward_fn = this->backward_fn;
            auto other_backward_fn = other.backward_fn;

            auto saved_this_shape = shape;      
            auto saved_other_shape = other.shape; 
            auto saved_batch_shape = batch_shape;

            result.backward_fn = [=]() mutable {
                size_t batch_size = 1;
                for (auto d : saved_batch_shape) batch_size *= d;

            
                // dA = dC * B^T
                if (this_requires_grad && this_grad) {
                    for (size_t b = 0; b < batch_size; b++) {
                        std::vector<size_t> batch_index = Tensor::unflatten_index(b, saved_batch_shape);
                        size_t A_offset = map_index(batch_index,
                                                    std::vector<size_t>(saved_this_shape.begin(),
                                                                        saved_this_shape.end() - 2))
                                          * (m * n);
                        size_t B_offset = map_index(batch_index,
                                                    std::vector<size_t>(saved_other_shape.begin(),
                                                                        saved_other_shape.end() - 2))
                                          * (n * p);
                        size_t C_offset = b * (m * p);

                        for (size_t i = 0; i < m; i++) {
                            for (size_t k = 0; k < n; k++) {
                                float grad_value = 0.0f;
                                for (size_t j = 0; j < p; j++) {
                                    grad_value += result_grad->data[C_offset + i*p + j]
                                                * B_data[B_offset + k*p + j];
                                }
                                this_grad->data[A_offset + i*n + k] += grad_value;
                            }
                        }
                    }
                    this_grad->data = reduce_grad(this_grad->data, saved_this_shape, saved_this_shape);
                
                    if (this_backward_fn) this_backward_fn();
                }

                // dB = A^T * dC
                if (other_requires_grad && other_grad) {
                    for (size_t b = 0; b < batch_size; b++) {
                        std::vector<size_t> batch_index = Tensor::unflatten_index(b, saved_batch_shape);
                        size_t A_offset = map_index(batch_index,
                                                    std::vector<size_t>(saved_this_shape.begin(),
                                                                        saved_this_shape.end() - 2))
                                          * (m * n);
                        size_t B_offset = map_index(batch_index,
                                                    std::vector<size_t>(saved_other_shape.begin(),
                                                                        saved_other_shape.end() - 2))
                                          * (n * p);
                        size_t C_offset = b * (m * p);

                        for (size_t k = 0; k < n; k++) {
                            for (size_t j = 0; j < p; j++) {
                                float grad_value = 0.0f;
                                for (size_t i = 0; i < m; i++) {
                                    grad_value += A_data[A_offset + i*n + k]
                                                * result_grad->data[C_offset + i*p + j];
                                }
                                other_grad->data[B_offset + k*p + j] += grad_value;
                            }
                        }
                    }
                    other_grad->data = reduce_grad(other_grad->data, saved_other_shape, saved_other_shape);
                    if (other_backward_fn) other_backward_fn();
                }
            };
        }
        
        // Return the final result
        return result;
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