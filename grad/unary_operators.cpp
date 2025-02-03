#include "cppgrad.h"

/* 
 * elementwise logarithm of the tensor
 * supports backpropagation
 */
Tensor Tensor::log() const {
    
    auto this_shape = this->ptr->shape;
    auto this_data = this->ptr->data;

    auto this_requires_grad = this->ptr->requires_grad;


    /* take the logarithm */
    std::vector<float> result_data(this_data.size());
    for (size_t i = 0; i < this_data.size(); i++) {
        result_data[i] = std::log(this_data[i]);
    }

    /* construct result tensor */
    Tensor result = Tensor(result_data, this_shape, this_requires_grad);

    /* construt backward function */
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

        result.ptr->backward_fn = [this_ptr = this->ptr, result_ptr = result.ptr]() {

            std::thread::id tid = std::this_thread::get_id();

            auto this_shape = this_ptr->shape;
            auto result_shape = result_ptr->shape;

            auto this_data = this_ptr->data;
            auto result_data = result_ptr->data;

            auto this_grad = this_ptr->thread_gradients[tid];
            auto result_grad = result_ptr->thread_gradients[tid]->data;

            if (this_ptr->requires_grad && this_grad) {
                for (size_t i = 0; i < this_data.size(); i++) {
                    this_grad->data[i] += result_grad[i] / this_data[i];
                }
            }
        };
    }

    return result;
}

/* 
 * elementwise exponential of the tensor
 * supports backpropagation
 */
Tensor Tensor::exp() const{

    auto this_shape = this->ptr->shape;
    auto this_data = this->ptr->data;

    auto this_requires_grad = this->ptr->requires_grad;

    /* take the logarithm */
    std::vector<float> result_data(this_data.size());
    for (size_t i = 0; i < this_data.size(); i++) {
        result_data[i] = std::exp(this_data[i]);
    }

    /* construct result tensor */
    Tensor result = Tensor(result_data, this_shape, this_requires_grad);
    
    /* construt backward function */
    if (result.ptr->requires_grad) {
        
        std::thread::id tid = std::this_thread::get_id();

        /* add result to computation graph */
        {
            std::lock_guard<std::mutex> lock(TensorData::GLOBAL_PARENTS_MUTEX);
            if (this_requires_grad) {
                result.ptr->parents[tid].insert(this->ptr);
            }

        }
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

        result.ptr->backward_fn = [this_ptr = this->ptr, result_ptr = result.ptr]() {

            std::thread::id tid = std::this_thread::get_id();

            auto result_shape = result_ptr->shape;

            auto this_data = this_ptr->data;
            auto result_data = result_ptr->data;

            auto this_grad = this_ptr->thread_gradients[tid];
            auto result_grad = result_ptr->thread_gradients[tid]->data;
            
            if (this_ptr->requires_grad && this_grad) {
                for (size_t i = 0; i < this_data.size(); i++) {
                    this_grad->data[i] += result_grad[i] * result_data[i];
                }
            }
        };
    }

    return result;
}