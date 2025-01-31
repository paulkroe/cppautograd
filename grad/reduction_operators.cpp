#include "cppgrad.h"

/*
 * reducing tensor by summing over given dimension
 * supports backpropagation
 */
Tensor Tensor::sum(size_t dim) const {
    /* check for valid dimension */
    if (dim >= shape.size()) {
        throw std::invalid_argument("Invalid dimension for sum operation");
    }

    /* compute new shape afer reduction */
    std::vector<size_t> new_shape = shape;
    new_shape.erase(new_shape.begin() + dim);

    /* allocate memory for result data */
    size_t result_size = numel(new_shape);
    std::vector<float> result_data(result_size, 0.0f);

    /* iterate over result data and perform addition */
    for (size_t i = 0; i < result_size; ++i) {
        
        /* compute the multi-dimensional index in the result shape */
        std::vector<size_t> coords_reduced = unravel_index(i, new_shape);

        std::vector<size_t> coords_full(shape.size());
        
        /* 
         * copy the reduced coords into coords_full, skipping an index for `dim`
         * I.e.:
         *   for i in [0..dim-1]: coords_full[i] = coords_reduced[i]
         *   coords_full[dim] = ...
         *   for i in [dim+1..end]: coords_full[i] = coords_reduced[i-1]
         */

        /* index for coords_reduced */
        size_t r_i = 0;
        for (size_t full_i = 0; full_i < shape.size(); full_i++) {
            if (full_i == dim) {
                /* skip entry in dim */
                continue;
            }
            coords_full[full_i] = coords_reduced[r_i];
            r_i++;
        }

        /* sum over j in [0..shape[dim]) */
        float sum_val = 0.0f;
        for (size_t j = 0; j < shape[dim]; j++) {
            /* index using j into left out dimension */
            coords_full[dim] = j;
            /* flatten out the multi-index */
            size_t index = ravel_index(coords_full, shape);
            sum_val += data[index];
        }

        /* insert sum in result tensor */
        result_data[i] = sum_val;
    }

    /* allocate result tensor */
    std::shared_ptr<Tensor> result = std::make_shared<Tensor>(result_data, new_shape, requires_grad);
    
    /* construct backward function */
    if (result->requires_grad) {
        /*
         * copy data necessary for backward function
         * to avoid dangling references
         */
        auto this_requires_grad = this->requires_grad;
        auto this_shape = this->shape;
        auto result_shape = result->shape;
        auto this_backward_fn = this->backward_fn;

        /* 
         * dimension that has been reduced,
         * needed to expand gradients during backpropagation
         */
        auto reduced_dim = dim;

        std::thread::id tid = std::this_thread::get_id();

        /* Store parents in a thread-safe manner */
        {
            std::lock_guard<std::mutex> lock(GLOBAL_PARENTS_MUTEX);
            if (this_requires_grad) {
                auto parent = std::make_shared<Tensor>(*this);
                parent->id = id; 
                result->parents[tid].insert(parent);
            }
        }

        /* Ensure thread-local gradients are initialized */
        std::shared_ptr<Tensor> this_grad, other_grad;
        {
            std::lock_guard<std::mutex> lock(GLOBAL_GRAD_MUTEX);
            if (this_requires_grad) {
                if (!this->thread_gradients[tid]) {
                    this->thread_gradients[tid] = std::make_shared<Tensor>(std::vector<float>(this->data.size(), 0.0f), this_shape, false);
                }
                this_grad = this->thread_gradients[tid];
            }
        }

        result->backward_fn = [
            this_requires_grad, this_grad, this_shape,
            result, result_shape, reduced_dim, this_backward_fn
        ]() {
            
            std::thread::id tid = std::this_thread::get_id();

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
            for (size_t i  = 0; i < result->thread_gradients[tid]->data.size(); ++i) {
                float grad_val = result->thread_gradients[tid]->data[i];

                /* coords_reduced in the result's shape */
                auto coords_reduced = unravel_index(i, result_shape);

                std::vector<size_t> coords_full(this_shape.size());
                
                /* fill coords_full except for the reduced_dim */
                size_t r_i = 0;
                for (size_t full_i = 0; full_i < this_shape.size(); full_i++) {
                    if (full_i == reduced_dim) {
                        continue;
                    }
                    coords_full[full_i] = coords_reduced[r_i];
                    r_i++;
                }

                /* spread the grad over the reduced dimension */
                for (size_t j = 0; j < this_shape[reduced_dim]; j++) {
                    coords_full[reduced_dim] = j;
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

    return *result;
}

/*
 * reducing tensor by summing over the tailing dimension
 * supports backpropagation
 */
Tensor Tensor::sum() const {
    return sum(shape.size() - 1);
}

/*
 * reducing tensor by taking mean over given dimension
 * supports backpropagation
 */
Tensor Tensor::mean(size_t dim) const {
    /* check for valid dimension */
    if (dim >= shape.size()) {
        throw std::invalid_argument("Invalid dimension for mean operation");
    }

    /* compute new shape afer reduction */
    std::vector<size_t> new_shape = shape;
    new_shape.erase(new_shape.begin() + dim);

    /* allocate memory for result data */
    size_t result_size = this->numel(new_shape);
    std::vector<float> result_data(result_size, 0.0f);

    /* divide by this factor after summation */
    float divisor = static_cast<float>(shape[dim]);

    /* iterate over result data and perform addition */
    for (size_t i = 0; i < result_size; ++i) {

        /* compute the multi-dimensional index in the result shape */
        std::vector<size_t> coords_reduced = unravel_index(i, new_shape);

        std::vector<size_t> coords_full(shape.size());
        
        /* 
         * copy the reduced coords into coords_full, skipping an index for `dim`
         * I.e.:
         *   for i in [0..dim-1]: coords_full[i] = coords_reduced[i]
         *   coords_full[dim] = ...
         *   for i in [dim+1..end]: coords_full[i] = coords_reduced[i-1]
         */

        /* index for coords_reduced */
        size_t r_i = 0; 
        for (size_t full_i = 0; full_i < shape.size(); full_i++) {
            if (full_i == dim) {
                /* skip entry in dim */
                continue;
            }
            coords_full[full_i] = coords_reduced[r_i];
            r_i++;
        }

        /* sum over j in [0..shape[dim]) */
        float sum_val = 0.0f;
        for (size_t j = 0; j < shape[dim]; j++) {
            /* index using j into left out dimension */
            coords_full[dim] = j;
            size_t index = ravel_index(coords_full, shape);
            sum_val += data[index];
        }

        /* insert sum in result tensor */
        result_data[i] = sum_val / divisor;
    }

    /* allocate result tensor */
    std::shared_ptr<Tensor> result = std::make_shared<Tensor>(result_data, new_shape, requires_grad);

    /* construct backward function */
    if (result->requires_grad) {
        /*
         * copy data necessary for backward function
         * to avoid dangling references
         */
        auto this_requires_grad = this->requires_grad;
        auto this_shape = this->shape;
        auto result_shape = result->shape;
        size_t reduced_dim = dim;
        auto this_backward_fn = this->backward_fn;
        float divisor_f = divisor;

        std::thread::id tid = std::this_thread::get_id();       

        /* Store parents in a thread-safe manner */
        {
            std::lock_guard<std::mutex> lock(GLOBAL_PARENTS_MUTEX);
            if (this_requires_grad) result->parents[tid].insert(std::make_shared<Tensor>(*this));
        }
        
        /* Ensure thread-local gradients are initialized */
        std::shared_ptr<Tensor> this_grad, other_grad;
        {
            std::lock_guard<std::mutex> lock(GLOBAL_GRAD_MUTEX);
            if (this_requires_grad) {
                if (!this->thread_gradients[tid]) {
                    this->thread_gradients[tid] = std::make_shared<Tensor>(std::vector<float>(this->data.size(), 0.0f), this_shape, false);
                }
                this_grad = this->thread_gradients[tid];
            }
        }

        result->backward_fn = [
            this_requires_grad, this_grad, this_shape,
            result, result_shape, reduced_dim, divisor_f, this_backward_fn
        ]() {
            
                std::thread::id tid = std::this_thread::get_id();

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
                for (size_t i = 0; i < result->thread_gradients[tid]->data.size(); ++i) {
                    float grad_val = result->thread_gradients[tid]->data[i];
                
                    /* coords_reduced in the result's shape */
                    auto coords_reduced = unravel_index(i, result_shape);

                    std::vector<size_t> coords_full(this_shape.size());
                    
                    /* fill coords_full except for the reduced_dim */
                    size_t r_i = 0; 
                    for (size_t full_i = 0; full_i < this_shape.size(); full_i++) {
                        if (full_i == reduced_dim) {
                            continue;
                        }
                        coords_full[full_i] = coords_reduced[r_i];
                        r_i++;
                    }

                    /* spread the grad over the reduced dimension */
                    for (size_t j = 0; j < this_shape[reduced_dim]; j++) {
                        coords_full[reduced_dim] = j;
                        size_t index = ravel_index(coords_full, this_shape);
                        expanded_grad[index] += grad_val / divisor_f;
                    }
                }

                /* now accumulate expanded_grad into this_grad->data */
                for (size_t i = 0; i < expanded_grad.size(); ++i) {
                    this_grad->data[i] += expanded_grad[i];
                }

        };
    }

    return *result;
}

/*
 * reducing tensor by taking the mean over the tailing dimension
 * supports backpropagation
 */
Tensor Tensor::mean() const {
    return mean(shape.size() - 1);
}