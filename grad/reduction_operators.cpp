#include "cppgrad.h"

/*
 * reducing tensor by summing over given dimension
 * supports backpropagation
 */
Tensor Tensor::sum(size_t dim) const {
    /* check for valid dimension */
    if (dim >= this->ptr->shape.size()) {
        throw std::invalid_argument("Invalid dimension for sum operation");
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

    /* iterate over result data and perform addition */
    for (size_t i = 0; i < result_size; ++i) {
        
        /* compute the multi-dimensional index in the result shape */
        std::vector<size_t> coords_reduced = unravel_index(i, new_shape);

        std::vector<size_t> coords_full(ptr->shape.size());
        
        /* 
         * copy the reduced coords into coords_full, skipping an index for `dim`
         * I.e.:
         *   for i in [0..dim-1]: coords_full[i] = coords_reduced[i]
         *   coords_full[dim] = ...
         *   for i in [dim+1..end]: coords_full[i] = coords_reduced[i-1]
         */

        /* index for coords_reduced */
        size_t r_i = 0;
        for (size_t full_i = 0; full_i < ptr->shape.size(); full_i++) {
            if (full_i == dim) {
                /* skip entry in dim */
                continue;
            }
            coords_full[full_i] = coords_reduced[r_i];
            r_i++;
        }

        /* sum over j in [0..shape[dim]) */
        float sum_val = 0.0f;
        for (size_t j = 0; j < ptr->shape[dim]; j++) {
            /* index using j into left out dimension */
            coords_full[dim] = j;
            /* flatten out the multi-index */
            size_t index = ravel_index(coords_full, ptr->shape);
            sum_val += ptr->data[index];
        }

        /* insert sum in result tensor */
        result_data[i] = sum_val;
    }

    /* allocate result tensor */
    Tensor result = Tensor(result_data, new_shape, ptr->requires_grad);
    
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
            if (this->ptr->requires_grad) {
                result.ptr->parents[tid].insert(this->ptr);
            }
        }

        /* Ensure thread-local gradients are initialized */
        std::shared_ptr<TensorData> this_grad;
        {
            std::lock_guard<std::mutex> lock(TensorData::GLOBAL_GRAD_MUTEX);
            if (this->ptr->requires_grad) {
                if (!this->ptr->thread_gradients[tid]) {
                    this->ptr->thread_gradients[tid] = std::make_shared<TensorData>(std::vector<float>(this->ptr->data.size(), 0.0f), this->ptr->shape, false);
                }
                this_grad = this->ptr->thread_gradients[tid];
            }
        }

        result.ptr->backward_fn = [
            this_ptr = this->ptr, result_ptr = result.ptr, dim]() {
            
            std::thread::id tid = std::this_thread::get_id();

            /* 
             * expand `result_grad` (shape = result_shape)
             * back to `this_shape`. 
             * for each element in the result_grad, we broadcast
             * over the reduced dimension.
             */
            auto this_grad = this_ptr->thread_gradients[tid];
            std::vector<float> expanded_grad(this_grad->data.size(), 0.0f);

            /*
             * for each index `i` in [0..result_grad->data.size()),
             * unravel it in the reduced (result's) shape
             * then for each `j` in [0..this_shape[reduced_dim]),
             * spread the gradient`
             */
            for (size_t i  = 0; i < result_ptr->thread_gradients[tid]->data.size(); ++i) {
                float grad_val = result_ptr->thread_gradients[tid]->data[i];

                /* coords_reduced in the result's shape */
                auto coords_reduced = unravel_index(i, result_ptr->shape);

                std::vector<size_t> coords_full(this_ptr->shape.size());
                
                /* fill coords_full except for the reduced_dim */
                size_t r_i = 0;
                for (size_t full_i = 0; full_i < this_ptr->shape.size(); full_i++) {
                    if (full_i == dim) {
                        continue;
                    }
                    coords_full[full_i] = coords_reduced[r_i];
                    r_i++;
                }

                /* spread the grad over the reduced dimension */
                for (size_t j = 0; j < this_ptr->shape[dim]; j++) {
                    coords_full[dim] = j;
                    /* flatten out the multi-index */
                    size_t index = ravel_index(coords_full, this_ptr->shape);
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

/*
 * reducing tensor by summing over the tailing dimension
 * supports backpropagation
 */
Tensor Tensor::sum() const {
    return sum(ptr->shape.size() - 1);
}

/*
 * reducing tensor by taking mean over given dimension
 * supports backpropagation
 */
Tensor Tensor::mean(size_t dim) const {
    /* check for valid dimension */
    if (dim >= ptr->shape.size()) {
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
    float divisor = static_cast<float>(ptr->shape[dim]);

    /* iterate over result data and perform addition */
    for (size_t i = 0; i < result_size; ++i) {

        /* compute the multi-dimensional index in the result shape */
        std::vector<size_t> coords_reduced = unravel_index(i, new_shape);

        std::vector<size_t> coords_full(ptr->shape.size());
        
        /* 
         * copy the reduced coords into coords_full, skipping an index for `dim`
         * I.e.:
         *   for i in [0..dim-1]: coords_full[i] = coords_reduced[i]
         *   coords_full[dim] = ...
         *   for i in [dim+1..end]: coords_full[i] = coords_reduced[i-1]
         */

        /* index for coords_reduced */
        size_t r_i = 0; 
        for (size_t full_i = 0; full_i < ptr->shape.size(); full_i++) {
            if (full_i == dim) {
                /* skip entry in dim */
                continue;
            }
            coords_full[full_i] = coords_reduced[r_i];
            r_i++;
        }

        /* sum over j in [0..shape[dim]) */
        float sum_val = 0.0f;
        for (size_t j = 0; j < ptr->shape[dim]; j++) {
            /* index using j into left out dimension */
            coords_full[dim] = j;
            size_t index = ravel_index(coords_full, ptr->shape);
            sum_val += ptr->data[index];
        }

        /* insert sum in result tensor */
        result_data[i] = sum_val / divisor;
    }

    /* allocate result tensor */
    Tensor result = Tensor(result_data, new_shape, this->ptr->requires_grad);

    /* construct backward function */
    if (result.ptr->requires_grad) {

        std::thread::id tid = std::this_thread::get_id();       

        /* add result to computation graph */
        {
            std::lock_guard<std::mutex> lock(TensorData::GLOBAL_PARENTS_MUTEX);
            if (this->ptr->requires_grad) {
                result.ptr->parents[tid].insert(this->ptr);
            }
        }
        
        /* Ensure thread-local gradients are initialized */
        std::shared_ptr<TensorData> this_grad;
        {
            std::lock_guard<std::mutex> lock(TensorData::GLOBAL_GRAD_MUTEX);
            if (this->ptr->requires_grad) {
                if (!this->ptr->thread_gradients[tid]) {
                    this->ptr->thread_gradients[tid] = std::make_shared<TensorData>(std::vector<float>(this->ptr->data.size(), 0.0f), this->ptr->shape, false);
                }
                this_grad = this->ptr->thread_gradients[tid];
            }
        }

        result.ptr->backward_fn = [this_ptr = this->ptr, result_ptr = result.ptr, dim, divisor]() {
            
                std::thread::id tid = std::this_thread::get_id();

               /* 
                * expand `result_grad` (shape = result_shape)
                * back to `this_shape`. 
                * for each element in the result_grad, we broadcast
                * over the reduced dimension.
                */
               auto this_grad = this_ptr->thread_gradients[tid];
               std::vector<float> expanded_grad(this_grad->data.size(), 0.0f);

               /*
                * for each index `i` in [0..result_grad->data.size()),
                * unravel it in the reduced (result's) shape
                * then for each `j` in [0..this_shape[reduced_dim]),
                * spread the gradient`
                */
                for (size_t i = 0; i < result_ptr->thread_gradients[tid]->data.size(); ++i) {
                    float grad_val = result_ptr->thread_gradients[tid]->data[i];
                
                    /* coords_reduced in the result's shape */
                    auto coords_reduced = unravel_index(i, result_ptr->shape);

                    std::vector<size_t> coords_full(this_ptr->shape.size());
                    
                    /* fill coords_full except for the reduced_dim */
                    size_t r_i = 0; 
                    for (size_t full_i = 0; full_i < this_ptr->shape.size(); full_i++) {
                        if (full_i == dim) {
                            continue;
                        }
                        coords_full[full_i] = coords_reduced[r_i];
                        r_i++;
                    }

                    /* spread the grad over the reduced dimension */
                    for (size_t j = 0; j < this_ptr->shape[dim]; j++) {
                        coords_full[dim] = j;
                        size_t index = ravel_index(coords_full, this_ptr->shape);
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

/*
 * reducing tensor by taking the mean over the tailing dimension
 * supports backpropagation
 */
Tensor Tensor::mean() const {
    return mean(ptr->shape.size() - 1);
}