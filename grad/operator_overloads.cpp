#include "cppgrad.h"

/**
 * @brief Element-wise addition of two tensors with broadcasting support.
 *
 * This operator performs element-wise addition between the current tensor and `other`,
 * supporting **broadcasting** to handle different shapes. The resulting tensor's shape
 * follows standard **NumPy broadcasting rules**.
 *
 * Steps:
 * 1. **Check broadcast compatibility:** Ensures the two tensors can be broadcasted.
 * 2. **Iterate over result data:** Performs element-wise addition with correct indexing.
 * 3. **Construct multi-dimensional indices:** Maps input indices to the broadcasted result.
 * 4. **Compute sum:** Stores the computed sum in the output tensor.
 * 5. **Set up autograd:** If either tensor requires gradients, registers a backward function.
 *
 * @param other The tensor to add to the current tensor.
 * @return Tensor The result of element-wise addition, with shape determined by broadcasting.
 * 
 * @throws std::invalid_argument If the tensors cannot be broadcasted.
 *
 * @note If either operand has `requires_grad = true`, the resulting tensor will also
 *       require gradients and track operations for automatic differentiation.
 *
 * @example
 * @code
 * Tensor a({1, 2, 3}, {3}, true);   // Shape: (3,)
 * Tensor b({5}, {1}, false);        // Shape: (1,)
 * Tensor c = a + b;                 // Shape: (3,), values: {6, 7, 8}
 * @endcode
 */
Tensor Tensor::operator+(const Tensor& other) const {

    
    auto this_shape = this->ptr->shape;
    auto other_shape = other.ptr->shape;

    auto this_data = this->ptr->data;
    auto other_data = other.ptr->data;

    auto this_requires_grad = this->ptr->requires_grad;
    auto other_requires_grad = other.ptr->requires_grad;
    auto result_requires_grad = this_requires_grad || other_requires_grad;


    /* infer result shape */
    std::vector<size_t> result_shape = broadcast(this_shape, other_shape);
    if (result_shape.empty()) {
        printShapes(this_shape, other_shape);
        throw std::invalid_argument("Tensor shapes must be broadcastable for addition.");
    }

    /* allocate memory for result data */
    size_t result_size = numel(result_shape);
    std::vector<float> result_data(result_size);

    /* iterate over result data and perform addition */
    for (size_t i = 0; i < result_size; ++i) {
        std::vector<size_t> multi_index = unravel_index(i, result_shape);
        size_t index_a = ravel_index(multi_index, this_shape);
        size_t index_b = ravel_index(multi_index, other_shape);
        result_data[i] = (this_data[index_a]) + other_data[index_b];
    }

    /* allocate result tensor */
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
            if (this->ptr->requires_grad) {
                if (!this->ptr->thread_gradients[tid]) {
                    this->ptr->thread_gradients[tid] = std::make_shared<TensorData>(std::vector<float>(this->ptr->data.size(), 0.0f), this_shape, false);
                }
                this_grad = this->ptr->thread_gradients[tid];
            }
        }
        {
            std::lock_guard<std::mutex> lock(TensorData::GLOBAL_GRAD_MUTEX);
            if (other.ptr->requires_grad) {
                if (!other.ptr->thread_gradients[tid]) {
                    other.ptr->thread_gradients[tid] = std::make_shared<TensorData>(std::vector<float>(other.ptr->data.size(), 0.0f), other_shape, false);
                }
                other_grad = other.ptr->thread_gradients[tid];
            }
        }

        /* Capture shared pointers to gradients in backward function */
        result.ptr->backward_fn = [this_ptr = this->ptr, other_ptr = other.ptr, result_ptr = result.ptr, result_shape = result.ptr->shape]() 
        {
            std::thread::id tid = std::this_thread::get_id();

            /* 1) Gradient w.r.t. x (this->data) */
            auto this_grad = this_ptr->thread_gradients[tid];
            if (this_ptr->requires_grad && this_grad) {
                auto reduced_grad = reduce_grad(result_ptr->thread_gradients[tid]->data, result_shape, this_ptr->shape);
                for (size_t i = 0; i < reduced_grad.size(); ++i) {
                    this_grad->data[i] += reduced_grad[i];
                }
            }

            /* 2) Gradient w.r.t. y (other->data) */
            auto other_grad = other_ptr->thread_gradients[tid];
            if (other_ptr->requires_grad && other_grad) {
                auto reduced_grad = reduce_grad(result_ptr->thread_gradients[tid]->data, result_shape, other_ptr->shape);
                for (size_t i = 0; i < reduced_grad.size(); ++i) {
                    other_grad->data[i] += reduced_grad[i];
                }
            }
        };
    }

    return result;
}

/**
 * @brief Adds a scalar value to each element of the tensor.
 *
 * This operator performs element-wise addition between the current tensor and a scalar.
 * It internally converts the scalar into a **broadcastable tensor** before performing the addition.
 *
 * @param other The scalar value to add to each element of the tensor.
 * @return Tensor The result of adding `other` to each element of the tensor.
 *
 * @note If the tensor has `requires_grad = true`, the resulting tensor will also
 *       require gradients and track operations for automatic differentiation.
 *
 * @example
 * @code
 * Tensor a({1.0, 2.0, 3.0}, {3}, true); // Shape: (3,)
 * Tensor b = a + 5.0f;                  // Shape: (3,), values: {6.0, 7.0, 8.0}
 * @endcode
 */
Tensor Tensor::operator+(const float other) const {
    return *this + Tensor({other}, false);
}

/**
 * @brief Negates all elements in the tensor.
 *
 * This operator applies element-wise negation to the tensor, returning a new tensor
 * where each element is multiplied by `-1`.
 *
 * Steps:
 * 1. **Negate the data:** Computes `-x` for each element.
 * 2. **Negate the gradient (if applicable):** Ensures correct gradient propagation.
 * 3. **Construct the result tensor:** Stores the negated values.
 * 4. **Set up autograd:** If `requires_grad = true`, registers a backward function.
 *
 * @return Tensor A tensor with all elements negated.
 *
 * @note If the tensor has `requires_grad = true`, the resulting tensor will also
 *       require gradients and track operations for automatic differentiation.
 *
 * @example
 * @code
 * Tensor a({1.0, -2.0, 3.0}, {3}, true); // Shape: (3,)
 * Tensor b = -a;                         // Shape: (3,), values: {-1.0, 2.0, -3.0}
 * @endcode
 */
Tensor Tensor::operator-() const {
    
    auto this_shape = this->ptr->shape;

    auto this_data = this->ptr->data;

    auto this_requires_grad = this->ptr->requires_grad;
    auto result_requires_grad = this_requires_grad;   

    /* negate data */
    std::vector<float> result_data(this_data.size());
    for (size_t i = 0; i < this_data.size(); i++) {
        result_data[i] = -this_data[i];
    }
    
    /* construct result tensor */
    Tensor result = Tensor(result_data, this_shape, result_requires_grad);

    /* construct backward function */
    if (result_requires_grad) {
        
        /* add result to computation graph */
        std::thread::id tid = std::this_thread::get_id();
        {
            std::lock_guard<std::mutex> lock(TensorData::GLOBAL_PARENTS_MUTEX);
            if (this_requires_grad) {
                result.ptr->parents[tid].insert(this->ptr);
            }
        }
        /* Ensure thread-local gradients are initialized */
        std::shared_ptr<TensorData> this_grad;
        {
            std::lock_guard<std::mutex> lock(TensorData::GLOBAL_GRAD_MUTEX);
            if (this_requires_grad) {
                if (!this->ptr->thread_gradients[tid]) {
                    this->ptr->thread_gradients[tid] = std::make_shared<TensorData>(std::vector<float>(this_data.size(), 0.0f), this_shape, false);
                }
                this_grad = this->ptr->thread_gradients[tid];
            }
        }

        result.ptr->backward_fn = [this_ptr = this->ptr, result_ptr = result.ptr]() 
     {
            std::thread::id tid = std::this_thread::get_id();

            auto this_data = this_ptr->data;
            auto this_grad = this_ptr->thread_gradients[tid];

            auto result_grad = result_ptr->thread_gradients[tid]->data;
            if (this_ptr->requires_grad && this_grad) {

                for (size_t i = 0; i < this_data.size(); i++) {
                    this_grad->data[i] -= result_grad[i];
                }
            }
        };
    }

    return result;
}

/**
 * @brief Performs element-wise subtraction between two tensors with broadcasting support.
 *
 * This operator computes the difference between the current tensor and `other`, supporting
 * **broadcasting** as per **NumPy broadcasting rules**. Internally, it utilizes the overloaded
 * unary negation and addition operators:
 * \f[
 * \text{result} = \text{this} - \text{other} = \text{this} + (-\text{other})
 * \f]
 *
 * @param other The tensor to subtract from the current tensor.
 * @return Tensor The result of element-wise subtraction, with shape determined by broadcasting.
 *
 * @throws std::invalid_argument If the tensors cannot be broadcasted.
 *
 * @note If either operand has `requires_grad = true`, the resulting tensor will also
 *       require gradients and track operations for automatic differentiation.
 *
 * @example
 * @code
 * Tensor a({1, 2, 3}, {3}, true);   // Shape: (3,)
 * Tensor b({5, 1, 0}, {3}, false);  // Shape: (3,)
 * Tensor c = a - b;                 // Shape: (3,), values: {-4, 1, 3}
 * @endcode
 */
Tensor Tensor::operator-(const Tensor& other) const{
    /* a - b =  a + (-b) */
    return *this + (-other);
}

/**
 * @brief Subtracts a scalar value from each element of the tensor.
 *
 * This operator performs element-wise subtraction between the tensor and a scalar.
 * Internally, it converts the scalar into a **broadcastable tensor** before performing the operation.
 *
 * @param other The scalar value to subtract from each element of the tensor.
 * @return Tensor The result of subtracting `other` from each element of the tensor.
 *
 * @note If the tensor has `requires_grad = true`, the resulting tensor will also
 *       require gradients and track operations for automatic differentiation.
 *
 * @example
 * @code
 * Tensor a({1.0, 2.0, 3.0}, {3}, true); // Shape: (3,)
 * Tensor b = a - 2.0f;                  // Shape: (3,), values: {-1.0, 0.0, 1.0}
 * @endcode
 */
Tensor Tensor::operator-(const float other) const {
    return *this - Tensor({other}, false);
}

/**
 * @brief Performs element-wise multiplication between two tensors with broadcasting support.
 *
 * This operator computes the product of the current tensor and `other`, supporting
 * **broadcasting** to handle different shapes as per **NumPy broadcasting rules**.
 *
 * Steps:
 * 1. **Check broadcast compatibility:** Ensures the two tensors can be broadcasted.
 * 2. **Iterate over result data:** Computes the product for each element.
 * 3. **Construct multi-dimensional indices:** Maps input indices to the broadcasted result.
 * 4. **Compute product:** Stores the computed product in the output tensor.
 * 5. **Set up autograd:** If either tensor requires gradients, registers a backward function.
 *
 * @param other The tensor to multiply with the current tensor.
 * @return Tensor The result of element-wise multiplication, with shape determined by broadcasting.
 *
 * @throws std::invalid_argument If the tensors cannot be broadcasted.
 *
 * @note If either operand has `requires_grad = true`, the resulting tensor will also
 *       require gradients and track operations for automatic differentiation.
 *
 * @example
 * @code
 * Tensor a({1, 2, 3}, {3}, true);   // Shape: (3,)
 * Tensor b({5, 1, 0}, {3}, false);  // Shape: (3,)
 * Tensor c = a * b;                 // Shape: (3,), values: {5, 2, 0}
 * @endcode
 */
Tensor Tensor::operator*(const Tensor& other) const {
    
    auto this_shape = this->ptr->shape;
    auto other_shape = other.ptr->shape;

    auto this_data = this->ptr->data;
    auto other_data = other.ptr->data;

    auto this_requires_grad = this->ptr->requires_grad;
    auto other_requires_grad = other.ptr->requires_grad;
    auto result_requires_grad = this_requires_grad || other_requires_grad;

    /* infer result shape */
    std::vector<size_t> result_shape;
    result_shape = broadcast(this_shape, other_shape);
    
    if (result_shape.empty()) {
        printShapes(this_shape, other_shape);
        throw std::invalid_argument("Tensor shapes must be broadcastable for multiplication.");
    }
            
    /* allocate memory for result data */
    size_t result_size = numel(result_shape);
    std::vector<float> result_data(result_size);

    /* iterate over result data and perform multiplication */
    for (size_t i = 0; i < result_size; ++i) {
        
        /* compute the multi-dimensional index in the result shape */
        std::vector<size_t> multi_index = unravel_index(i, result_shape);

        /* map to indices in the original tensors */
        size_t index_a = ravel_index(multi_index, this_shape);
        size_t index_b = ravel_index(multi_index, other_shape);

        /* perform addition */
        result_data[i] = this_data[index_a] * other_data[index_b];
    }

    /* allocate result tensor */
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
                    this->ptr->thread_gradients[tid] = std::make_shared<TensorData>(std::vector<float>(this->ptr->data.size(), 0.0f), this_shape, false);
                }
                this_grad = this->ptr->thread_gradients[tid];
            }
        }
        {
            std::lock_guard<std::mutex> lock(TensorData::GLOBAL_GRAD_MUTEX);
            if (other_requires_grad) {
                if (!other.ptr->thread_gradients[tid]) {
                    other.ptr->thread_gradients[tid] = std::make_shared<TensorData>(std::vector<float>(other.ptr->data.size(), 0.0f), other_shape, false);
                }
                other_grad = other.ptr->thread_gradients[tid];
            }
        }

        result.ptr->backward_fn = [this_ptr = this->ptr, other_ptr = other.ptr, result_ptr = result.ptr]() 
        {
            std::thread::id tid = std::this_thread::get_id();

            auto this_shape = this_ptr->shape;
            auto other_shape = other_ptr->shape;

            auto this_data = this_ptr->data;
            auto other_data = other_ptr->data;

            auto result_data = result_ptr->data;
            auto result_grad = result_ptr->thread_gradients[tid]->data;

            /* 1) Gradient w.r.t. x (this->data) */
            auto this_grad = this_ptr->thread_gradients[tid];
            if (this_ptr->requires_grad && this_grad) {

                /* temporary helper varibale to element wise gradient */
                std::vector<float> partial_grad_x(result_grad.size(), 0.0f);

                /* iterate over result data and compute gradient */
                for (size_t i = 0; i < result_grad.size(); ++i) {
                    /* compute the multi-dimensional index in the result shape */
                    std::vector<size_t> multi_index = unravel_index(i, result_ptr->shape);
                    /* map index into other->data */
                    size_t index_b = ravel_index(multi_index, other_shape);
                    /* compute the gradient */
                    partial_grad_x[i] = result_grad[i] * other_data[index_b];
                }

                /* reduce the gradient by summing over broadcasted dimension */
                auto reduced_grad_x = reduce_grad(partial_grad_x, result_ptr->shape, this_shape);
                
                for (size_t i = 0; i < reduced_grad_x.size(); ++i) {
                    this_grad->data[i] += reduced_grad_x[i];
                }

            } 

            /* 2) Gradient w.r.t. y (other->data) */
            auto other_grad = other_ptr->thread_gradients[tid];
            if (other_ptr->requires_grad && other_grad) {
                /* temporary helper varibale to element wise gradient */
                std::vector<float> partial_grad_y(result_grad.size(), 0.0f);

                /* iterate over result data and compute gradient */
                for (size_t i = 0; i < result_grad.size(); i++) {
                    /* compute the multi-dimensional index in the result shape */
                    std::vector<size_t> multi_index = unravel_index(i, result_ptr->shape);
                    /* map index into this->data */
                    size_t index_a = ravel_index(multi_index, this_shape);
                    /* compute the gradient */
                    partial_grad_y[i] = result_grad[i] * this_data[index_a];
                }
                
                /* reduce the gradient by summing over broadcasted dimension */
                auto reduced_grad_y = reduce_grad(partial_grad_y, result_ptr->shape, other_shape);
                
                for (size_t i = 0; i < reduced_grad_y.size(); ++i) {
                    other_grad->data[i] += reduced_grad_y[i];
                }
            }
        };
    }

    return result;
}

/**
 * @brief Multiplies each element of the tensor by a scalar value.
 *
 * This operator performs element-wise multiplication between the tensor and a scalar.
 * Internally, it converts the scalar into a **broadcastable tensor** before performing
 * the operation.
 *
 * @param other The scalar value to multiply with each element of the tensor.
 * @return Tensor The result of multiplying `other` with each element of the tensor.
 *
 * @note If the tensor has `requires_grad = true`, the resulting tensor will also
 *       require gradients and track operations for automatic differentiation.
 *
 * @example
 * @code
 * Tensor a({1.0, 2.0, 3.0}, {3}, true); // Shape: (3,)
 * Tensor b = a * 2.0f;                  // Shape: (3,), values: {2.0, 4.0, 6.0}
 * @endcode
 */
Tensor Tensor::operator*(const float other) const {
    return *this * Tensor({other}, false);
}

/**
 * @brief Performs element-wise division between two tensors with broadcasting support.
 *
 * This operator computes the division of the current tensor by `other`, supporting
 * **broadcasting** as per **NumPy broadcasting rules**.
 *
 * Steps:
 * 1. **Check broadcast compatibility:** Ensures the two tensors can be broadcasted.
 * 2. **Iterate over result data:** Computes the quotient for each element.
 * 3. **Construct multi-dimensional indices:** Maps input indices to the broadcasted result.
 * 4. **Compute division:** Stores the computed quotient in the output tensor.
 * 5. **Set up autograd:** If either tensor requires gradients, registers a backward function.
 *
 * @param other The tensor by which to divide the current tensor.
 * @return Tensor The result of element-wise division, with shape determined by broadcasting.
 *
 * @throws std::invalid_argument If the tensors cannot be broadcasted.
 * @throws std::runtime_error If division by zero occurs.
 *
 * @note If either operand has `requires_grad = true`, the resulting tensor will also
 *       require gradients and track operations for automatic differentiation.
 *
 * @example
 * @code
 * Tensor a({10, 20, 30}, {3}, true);  // Shape: (3,)
 * Tensor b({2, 5, 3}, {3}, false);    // Shape: (3,)
 * Tensor c = a / b;                   // Shape: (3,), values: {5.0, 4.0, 10.0}
 * @endcode
 */
Tensor Tensor::operator/(const Tensor& other) const {
 
    auto this_shape = this->ptr->shape;
    auto other_shape = other.ptr->shape;

    auto this_data = this->ptr->data;
    auto other_data = other.ptr->data;

    auto this_requires_grad = this->ptr->requires_grad;
    auto other_requires_grad = other.ptr->requires_grad;
    auto result_requires_grad = this_requires_grad || other_requires_grad;   
   
    /* infer result shape */
    std::vector<size_t> result_shape;
    result_shape = broadcast(this_shape, other_shape);
    
    if (result_shape.empty()) {
        printShapes(this_shape, other_shape);
        throw std::invalid_argument("Tensor shapes must be broadcastable for division.");
    }
            
    /* allocate memory for result data */
    size_t result_size = numel(result_shape);
    std::vector<float> result_data(result_size);

    /* iterate over result data and perform division */
    for (size_t i = 0; i < result_size; ++i) {

        /* compute the multi-dimensional index in the result shape */
        std::vector<size_t> multi_index = unravel_index(i, result_shape);

        /* map to indices in the original tensors */
        size_t index_a = ravel_index(multi_index, this_shape);
        size_t index_b = ravel_index(multi_index, other_shape);

        /* Perform the division */
        result_data[i] = this_data[index_a] / other_data[index_b];
    }

    /* allocate result tensor */
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
                    this->ptr->thread_gradients[tid] = std::make_shared<TensorData>(std::vector<float>(this->ptr->data.size(), 0.0f), this_shape, false);
                }
                this_grad = this->ptr->thread_gradients[tid];
            }
        }
        {
            std::lock_guard<std::mutex> lock(TensorData::GLOBAL_GRAD_MUTEX);
            if (other_requires_grad) {
                if (!other.ptr->thread_gradients[tid]) {
                    other.ptr->thread_gradients[tid] = std::make_shared<TensorData>(std::vector<float>(other.ptr->data.size(), 0.0f), other_shape, false);
                }
                other_grad = other.ptr->thread_gradients[tid];
            }
        }

        result.ptr->backward_fn = [this_ptr = this->ptr, other_ptr = other.ptr, result_ptr = result.ptr]() 
        {

            std::thread::id tid = std::this_thread::get_id();

            auto this_shape = this_ptr->shape;
            auto other_shape = other_ptr->shape;

            auto this_data = this_ptr->data;
            auto other_data = other_ptr->data;

            auto result_shape = result_ptr->shape;
            auto result_data = result_ptr->data;
            auto result_grad = result_ptr->thread_gradients[tid]->data;

            auto this_grad = this_ptr->thread_gradients[tid];
            auto other_grad = other_ptr->thread_gradients[tid];
            /* 1) Gradient w.r.t. x (this->data) */
            if (this_ptr->requires_grad && this_grad)
            {
                /* temporary helper varibale to element wise gradient */
                std::vector<float> partial_grad_x(result_grad.size(), 0.0f);

                /* iterate over result data and compute gradient */
                for (size_t i = 0; i < result_grad.size(); ++i) {
                    /* compute the multi-dimensional index in the result shape */
                    std::vector<size_t> multi_index = unravel_index(i, result_shape); 
                    /* map index into other->data */
                    size_t index_b = ravel_index(multi_index, other_shape);
                    /* compute the gradient */
                    partial_grad_x[i] = result_grad[i] / other_data[index_b];
                }

                /* reduce the gradient by summing over broadcasted dimension */
                auto reduced_grad_x = reduce_grad(partial_grad_x, result_shape, this_shape);
                
                for (size_t i = 0; i < reduced_grad_x.size(); ++i) {
                    this_grad->data[i] += reduced_grad_x[i];
                }
            } 

            /* 2) Gradient w.r.t. y (other->data) */
            if (other_ptr->requires_grad && other_grad) {
                /* temporary helper varibale to element wise gradient */
                std::vector<float> partial_grad_y(result_grad.size(), 0.0f);

                /* iterate over result data and compute gradient */
                for (size_t i = 0; i < result_grad.size(); i++) {
                    /* compute the multi-dimensional index in the result shape */
                    std::vector<size_t> multi_index = unravel_index(i, result_shape);
                    /* map index into data */
                    size_t index_a = ravel_index(multi_index, this_shape);
                    size_t index_b = ravel_index(multi_index, other_shape);
                    /* compute the gradient */
                    partial_grad_y[i] = -result_grad[i] * this_data[index_a] / (other_data[index_b] * other_data[index_b]);
                }

                /* reduce the gradient by summing over broadcasted dimension */
                auto reduced_grad_y = reduce_grad(partial_grad_y, result_shape, other_shape);
                
                for (size_t i = 0; i < reduced_grad_y.size(); ++i) {
                    other_grad->data[i] += reduced_grad_y[i];
                }
            }
        };
    }

    return result;
}

/**
 * @brief Divides each element of the tensor by a scalar value.
 *
 * This operator performs element-wise division between the tensor and a scalar.
 * Internally, it converts the scalar into a **broadcastable tensor** before performing
 * the operation.
 *
 * @param other The scalar value by which to divide each element of the tensor.
 * @return Tensor The result of dividing each element of the tensor by `other`.
 *
 * @throws std::runtime_error If division by zero occurs.
 *
 * @note If the tensor has `requires_grad = true`, the resulting tensor will also
 *       require gradients and track operations for automatic differentiation.
 *
 * @example
 * @code
 * Tensor a({10.0, 20.0, 30.0}, {3}, true); // Shape: (3,)
 * Tensor b = a / 2.0f;                     // Shape: (3,), values: {5.0, 10.0, 15.0}
 * @endcode
 */
Tensor Tensor::operator/(const float other) const {
    return *this / Tensor({other}, false);
}

/**
 * @brief Recursively prints tensor data in a structured format.
 *
 * This function is used internally by the `<<` operator to print tensor values
 * in a **multi-dimensional array format**. It ensures that nested dimensions
 * are displayed correctly.
 *
 * @param os The output stream to write to.
 * @param dim The current dimension being processed.
 * @param offset The starting index in the tensor's data vector.
 * @param stride The step size to traverse along the current dimension.
 *
 * @example
 * @code
 * Tensor t({1.0, 2.0, 3.0, 4.0}, {2, 2});
 * std::cout << t; // Calls print_recursive() internally
 * @endcode
 */
void Tensor::print_recursive(std::ostream& os, size_t dim, size_t offset, size_t stride) const {
    os << "[";
    for (size_t i = 0; i < ptr->shape[dim]; ++i) {
        if (dim == ptr->shape.size() - 1) { 
            /* last dimension, print values directly */
            os << ptr->data[offset + i];
        } else {
            /* recursively print nested dimensions */
            print_recursive(os, dim + 1, offset + i * stride, stride / ptr->shape[dim + 1]);
        }
        if (i + 1 < ptr->shape[dim]) os << ", ";
    }
    os << "]";
}

/**
 * @brief Overloaded output stream operator for printing tensors.
 *
 * This operator prints the tensor in a format similar to PyTorch:
 * ```
 * Tensor([[1.0, 2.0], [3.0, 4.0]], shape=[2, 2])
 * ```
 * The tensor values are printed using `print_recursive()` to ensure correct formatting
 * for multi-dimensional tensors.
 *
 * @param os The output stream to write to.
 * @param tensor The tensor to be printed.
 * @return std::ostream& The output stream with the formatted tensor data.
 *
 * @example
 * @code
 * Tensor t({1.0, 2.0, 3.0, 4.0}, {2, 2});
 * std::cout << t; // Prints: Tensor([[1.0, 2.0], [3.0, 4.0]], shape=[2, 2])
 * @endcode
 */
std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    os << "Tensor(";
    tensor.print_recursive(os, 0, 0, tensor.ptr->data.size() / tensor.ptr->shape[0]);
    os << ", shape=";
    os << "[";
    for (size_t i = 0; i < tensor.ptr->shape.size(); ++i) {
        os << tensor.ptr->shape[i];
        if (i + 1 < tensor.ptr->shape.size()) os << ", ";
    }
    os << "]";
    os << ")";
    return os;
}