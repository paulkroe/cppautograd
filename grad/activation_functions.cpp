#include "cppgrad.h"
Tensor Tensor::relu() const{
    
    /* apply the relu */
    std::vector<float> result_data(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        result_data[i] = data[i] > 0 ? data[i] : 0;
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

        std::thread::id tid = std::this_thread::get_id();

        /* add result to computation graph */
        {
            std::lock_guard<std::mutex> lock(GLOBAL_PARENTS_MUTEX);
            if (this_requires_grad) {
                auto parent = std::make_shared<Tensor>(*this);
                /* should be the same tensor in the computation graph */
                parent->id = this->id;
                result->parents[tid].insert(parent);
            }
        }

        /* ensure thread-local gradients are initialized */
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
            this_data, result
        ]() {

            std::thread::id tid = std::this_thread::get_id();

            if (this_requires_grad && this_grad) {
                for (size_t i = 0; i < this_data.size(); i++) {
                    this_grad->data[i] += result->thread_gradients[tid]->data[i] * (this_data[i] > 0 ? 1 : 0);
                }
            }
        };
    }

    return *result;
}