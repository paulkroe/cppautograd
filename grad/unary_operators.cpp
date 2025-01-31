#include "cppgrad.h"

/* 
 * elementwise logarithm of the tensor
 * supports backpropagation
 */
Tensor Tensor::log() const {
    
    /* take the logarithm */
    std::vector<float> result_data(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        result_data[i] = std::log(data[i]);
    }

    /* construct result tensor */
    std::shared_ptr<Tensor> result = std::make_shared<Tensor>(result_data, shape, requires_grad);

    /* construt backward function */
    if (result->requires_grad) {
        
        /*
         * copy data necessary for backward function
         * to avoid dangling references
         */
        auto this_requires_grad = this->requires_grad;
        auto this_data    = this->data;
        auto out_data     = result->data;

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
                    this->thread_gradients[tid] = std::make_shared<Tensor>(std::vector<float>(this->data.size(), 0.0f), this->shape, false);
                }
                this_grad = this->thread_gradients[tid];
            }
        }

        result->backward_fn = [
            this_requires_grad, this_grad,
            this_data, out_data, result
        ]() {

            std::thread::id tid = std::this_thread::get_id();

            if (this_requires_grad && this_grad) {
                for (size_t i = 0; i < this_data.size(); i++) {
                    this_grad->data[i] += result->thread_gradients[tid]->data[i] / this_data[i];
                }
            }
        };
    }

    return *result;
}

/* 
 * elementwise exponential of the tensor
 * supports backpropagation
 */
Tensor Tensor::exp() const{

    /* take the logarithm */
    std::vector<float> forward_data(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        forward_data[i] = std::exp(data[i]);
    }

    /* construct result tensor */
    std::shared_ptr<Tensor> result = std::make_shared<Tensor>(forward_data, shape, requires_grad);
    
    /* construt backward function */
    if (result->requires_grad) {
        
        /*
         * copy data necessary for backward function
         * to avoid dangling references
         */
        auto this_requires_grad = this->requires_grad;
        auto this_data    = this->data;

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
        std::shared_ptr<Tensor> this_grad;
        {
            std::lock_guard<std::mutex> lock(GLOBAL_GRAD_MUTEX);
            if (this_requires_grad) {
                if (!this->thread_gradients[tid]) {
                    this->thread_gradients[tid] = std::make_shared<Tensor>(std::vector<float>(this->data.size(), 0.0f), this->shape, false);
                }
                this_grad = this->thread_gradients[tid];
            }
        }

        result->backward_fn = [
            this_requires_grad, this_grad,
            this_data, result
        ]() {
            std::thread::id tid = std::this_thread::get_id();
            
            if (this_requires_grad && this_grad) {
                for (size_t i = 0; i < this_data.size(); i++) {
                    this_grad->data[i] += result->thread_gradients[tid]->data[i] * result->data[i];
                }
            }
        };
    }

    return *result;
}