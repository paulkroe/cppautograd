#include "cppgrad.h"

/**
 * @brief Computes the sum of the tensor elements along a given dimension.
 *
 * This function reduces the tensor by summing over the specified dimension,
 * producing a new tensor with one less dimension.
 * 
 * It supports **automatic differentiation**, meaning the sum operation 
 * contributes to backpropagation.
 *
 * @param dim The dimension along which to compute the sum.
 * @return Tensor A reduced tensor with one less dimension.
 *
 * @throws std::invalid_argument If `dim` is out of bounds.
 *
 * @note The resulting tensor retains `requires_grad` if the original tensor does.
 *
 * @example
 * @code
 * Tensor t({{1.0, 2.0}, {3.0, 4.0}}, {2, 2}, true); // Shape: (2,2)
 * Tensor s = t.sum(0); // Shape: (1,2), values: {4.0, 6.0}
 * @endcode
 */
Tensor Tensor::sum(size_t dim) const {

    auto this_shape = ptr->shape;
    auto this_data = ptr->data;
    auto this_requires_grad = ptr->requires_grad;
    
    /* check for valid dimension */
    if (dim >= this_shape.size()) {
        throw std::invalid_argument("Invalid dimension for sum operation");
    }

    /* compute new shape afer reduction */
    std::vector<size_t> new_shape = this_shape;
    new_shape.erase(new_shape.begin() + dim);
    if (new_shape.empty()) {
        new_shape.push_back(1);
    }

    /* allocate memory for result data */
    size_t result_size = numel(new_shape);
    std::vector<float> result_data(result_size, 0.0f);

    /* iterate over result data and perform addition */
    for (size_t i = 0; i < result_size; ++i) {
        
        /* compute the multi-dimensional index in the result shape */
        std::vector<size_t> coords_reduced = unravel_index(i, new_shape);

        std::vector<size_t> coords_full(this_shape.size());
        
        /* 
         * copy the reduced coords into coords_full, skipping an index for `dim`
         * I.e.:
         *   for i in [0..dim-1]: coords_full[i] = coords_reduced[i]
         *   coords_full[dim] = ...
         *   for i in [dim+1..end]: coords_full[i] = coords_reduced[i-1]
         */

        /* index for coords_reduced */
        size_t r_i = 0;
        for (size_t full_i = 0; full_i < this_shape.size(); full_i++) {
            if (full_i == dim) {
                /* skip entry in dim */
                continue;
            }
            coords_full[full_i] = coords_reduced[r_i];
            r_i++;
        }

        /* sum over j in [0..shape[dim]) */
        float sum_val = 0.0f;
        for (size_t j = 0; j < this_shape[dim]; j++) {
            /* index using j into left out dimension */
            coords_full[dim] = j;
            /* flatten out the multi-index */
            size_t index = ravel_index(coords_full, this_shape);
            sum_val += this_data[index];
        }

        /* insert sum in result tensor */
        result_data[i] = sum_val;
    }

    /* allocate result tensor */
    Tensor result = Tensor(result_data, new_shape, this_requires_grad);
    
    /* construct backward function */
    if (result.ptr->requires_grad) {

        /* 
         * dimension that has been reduced,
         * needed to expand gradients during backpropagation
         */

        std::thread::id tid = std::this_thread::get_id();

        /* add result to computation graph */
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
            if (this->ptr->requires_grad) {
                if (!this->ptr->thread_gradients[tid]) {
                    this->ptr->thread_gradients[tid] = std::make_shared<TensorData>(std::vector<float>(this_data.size(), 0.0f), this_shape, false);
                }
                this_grad = this->ptr->thread_gradients[tid];
            }
        }

        result.ptr->backward_fn = [
            this_ptr = this->ptr, this_shape, result_ptr = result.ptr, result_shape = new_shape, dim]() {
            
            std::thread::id tid = std::this_thread::get_id();

            auto this_grad = this_ptr->thread_gradients[tid];
            auto result_grad = result_ptr->thread_gradients[tid]->data;

            /* 
             * expand `result_grad` (shape = result_shape)
             * back to `this_shape`. 
             * for each element in the result_grad, we broadcast
             * over the reduced dimension.
             */
            std::vector<float> expanded_grad(this_grad->data.size(), 0.0f);

            /*
             * for each index `i` in [0..result_grad->data.size()),
             * unravel it in the reduced (result's) shape
             * then for each `j` in [0..this_shape[reduced_dim]),
             * spread the gradient`
             */
            for (size_t i  = 0; i < result_grad.size(); ++i) {
                float grad_val = result_grad[i];

                /* coords_reduced in the result's shape */
                auto coords_reduced = unravel_index(i, result_shape);

                std::vector<size_t> coords_full(this_shape.size());
                
                /* fill coords_full except for the reduced_dim */
                size_t r_i = 0;
                for (size_t full_i = 0; full_i < this_shape.size(); full_i++) {
                    if (full_i == dim) {
                        continue;
                    }
                    coords_full[full_i] = coords_reduced[r_i];
                    r_i++;
                }

                /* spread the grad over the reduced dimension */
                for (size_t j = 0; j < this_shape[dim]; j++) {
                    coords_full[dim] = j;
                    /* flatten out the multi-index */
                    size_t index = ravel_index(coords_full, this_shape);
                    expanded_grad[index] += grad_val;
                }
            }

            /* now accumulate expanded_grad into this_grad->data */
            for (size_t i = 0; i < expanded_grad.size(); ++i) {
                this_grad->data[i] += expanded_grad[i];
            }

        };
    }

    return result;
}

/**
 * @brief Computes the sum over the last dimension of the tensor.
 *
 * This is a convenience function that calls `sum(dim)`, where `dim`
 * is the last axis (`shape.size() - 1`). It reduces the tensor by summing
 * along the trailing dimension.
 *
 * @return Tensor A reduced tensor with one less dimension.
 *
 * @throws std::invalid_argument If the tensor has no dimensions.
 *
 * @note Supports **automatic differentiation** and retains `requires_grad` if applicable.
 *
 * @example
 * @code
 * Tensor t({{1.0, 2.0}, {3.0, 4.0}}, {2, 2}, true);
 * Tensor s = t.sum(); // Shape: (2,), values: {3.0, 7.0}
 * @endcode
 */
Tensor Tensor::sum() const {
    return sum(ptr->shape.size() - 1);
}

/**
 * @brief Computes the mean of the tensor elements along a given dimension.
 *
 * This function reduces the tensor by computing the average over the specified dimension,
 * producing a new tensor with one less dimension.
 *
 * It supports **automatic differentiation**, allowing gradients to propagate correctly.
 *
 * @param dim The dimension along which to compute the mean.
 * @return Tensor A reduced tensor with one less dimension.
 *
 * @throws std::invalid_argument If `dim` is out of bounds.
 *
 * @note The resulting tensor retains `requires_grad` if the original tensor does.
 *
 * @example
 * @code
 * Tensor t({{1.0, 2.0}, {3.0, 4.0}}, {2, 2}, true); // Shape: (2,2)
 * Tensor m = t.mean(0); // Shape: (1,2), values: {2.0, 3.0}
 * @endcode
 */
Tensor Tensor::mean(size_t dim) const {

    auto this_shape = ptr->shape;
    auto this_data = ptr->data;
    auto this_requires_grad = ptr->requires_grad;

    /* check for valid dimension */
    if (dim >= this_shape.size()) {
        throw std::invalid_argument("Invalid dimension for mean operation");
    }

    /* compute new shape afer reduction */
    std::vector<size_t> new_shape = ptr->shape;
    new_shape.erase(new_shape.begin() + dim);

    if (new_shape.empty()) {
        new_shape.push_back(1);
    }

    /* allocate memory for result data */
    size_t result_size = numel(new_shape);
    std::vector<float> result_data(result_size, 0.0f);

    /* divide by this factor after summation */
    float divisor = static_cast<float>(this_shape[dim]);

    /* iterate over result data and perform addition */
    for (size_t i = 0; i < result_size; ++i) {

        /* compute the multi-dimensional index in the result shape */
        std::vector<size_t> coords_reduced = unravel_index(i, new_shape);

        std::vector<size_t> coords_full(this_shape.size());
        
        /* 
         * copy the reduced coords into coords_full, skipping an index for `dim`
         * I.e.:
         *   for i in [0..dim-1]: coords_full[i] = coords_reduced[i]
         *   coords_full[dim] = ...
         *   for i in [dim+1..end]: coords_full[i] = coords_reduced[i-1]
         */

        /* index for coords_reduced */
        size_t r_i = 0; 
        for (size_t full_i = 0; full_i < this_shape.size(); full_i++) {
            if (full_i == dim) {
                /* skip entry in dim */
                continue;
            }
            coords_full[full_i] = coords_reduced[r_i];
            r_i++;
        }

        /* sum over j in [0..shape[dim]) */
        float sum_val = 0.0f;
        for (size_t j = 0; j < this_shape[dim]; j++) {
            /* index using j into left out dimension */
            coords_full[dim] = j;
            size_t index = ravel_index(coords_full, this_shape);
            sum_val += ptr->data[index];
        }

        /* insert sum in result tensor */
        result_data[i] = sum_val / divisor;
    }

    /* allocate result tensor */
    Tensor result = Tensor(result_data, new_shape, this_requires_grad);

    /* construct backward function */
    if (result.ptr->requires_grad) {

        std::thread::id tid = std::this_thread::get_id();       

        /* add result to computation graph */
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

        result.ptr->backward_fn = [this_ptr = this->ptr, this_shape, result_ptr = result.ptr, result_shape = new_shape, dim, divisor]() {
            
                std::thread::id tid = std::this_thread::get_id();

               auto this_grad = this_ptr->thread_gradients[tid];
               auto result_grad = result_ptr->thread_gradients[tid]->data;
               /* 
                * expand `result_grad` (shape = result_shape)
                * back to `this_shape`. 
                * for each element in the result_grad, we broadcast
                * over the reduced dimension.
                */
               
               std::vector<float> expanded_grad(this_grad->data.size(), 0.0f);

               /*
                * for each index `i` in [0..result_grad->data.size()),
                * unravel it in the reduced (result's) shape
                * then for each `j` in [0..this_shape[reduced_dim]),
                * spread the gradient`
                */
                for (size_t i = 0; i < result_grad.size(); ++i) {
                    float grad_val = result_grad[i];
                
                    /* coords_reduced in the result's shape */
                    auto coords_reduced = unravel_index(i, result_shape);

                    std::vector<size_t> coords_full(this_shape.size());
                    
                    /* fill coords_full except for the reduced_dim */
                    size_t r_i = 0; 
                    for (size_t full_i = 0; full_i < this_shape.size(); full_i++) {
                        if (full_i == dim) {
                            continue;
                        }
                        coords_full[full_i] = coords_reduced[r_i];
                        r_i++;
                    }

                    /* spread the grad over the reduced dimension */
                    for (size_t j = 0; j < this_shape[dim]; j++) {
                        coords_full[dim] = j;
                        size_t index = ravel_index(coords_full, this_shape);
                        expanded_grad[index] += grad_val / divisor;
                    }
                }

                /* now accumulate expanded_grad into this_grad->data */
                for (size_t i = 0; i < expanded_grad.size(); ++i) { 
                    this_grad->data[i] += expanded_grad[i];
                }

        };
    }

    return result;
}

/**
 * @brief Computes the mean over the last dimension of the tensor.
 *
 * This is a convenience function that calls `mean(dim)`, where `dim`
 * is the last axis (`shape.size() - 1`). It reduces the tensor by averaging
 * along the trailing dimension.
 *
 * @return Tensor A reduced tensor with one less dimension.
 *
 * @throws std::invalid_argument If the tensor has no dimensions.
 *
 * @note Supports **automatic differentiation** and retains `requires_grad` if applicable.
 *
 * @example
 * @code
 * Tensor t({{1.0, 2.0}, {3.0, 4.0}}, {2, 2}, true);
 * Tensor m = t.mean(); // Shape: (2,), values: {1.5, 3.5}
 * @endcode
 */
Tensor Tensor::mean() const {

    return mean(ptr->shape.size() - 1);
}