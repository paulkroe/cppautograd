#include "cppgrad.h"
/**
 * @brief Applies the ReLU (Rectified Linear Unit) activation function element-wise.
 *
 * This function computes the ReLU activation for each element in the tensor,
 * defined as:
 * \f[
 * f(x) = \max(0, x)
 * \f]
 *
 * If the input tensor has `requires_grad` set to `true`, this function:
 * - Constructs a computation graph entry for automatic differentiation.
 * - Stores a backward function that computes the derivative of ReLU, which is
 *   `1` for positive inputs and `0` for non-positive inputs.
 * - Ensures per-thread gradient tracking for multi-threaded execution.
 *
 * @return Tensor The result tensor with the same shape as the input tensor.
 * 
 * @note If `requires_grad` is `true`, this function registers the tensor in the computation graph
 *       and initializes thread-local gradients.
 * 
 * @example Usage:
 * @code
 * Tensor x({-2.0f, 3.0f, -1.0f, 4.0f}, {2, 2}, true);
 * Tensor y = x.relu();
 * @endcode
 */
Tensor Tensor::relu() const {
    
    auto this_shape = this->ptr->shape;
    auto this_data = this->ptr->data;

    auto this_requires_grad = this->ptr->requires_grad;
    auto result_requires_grad = this_requires_grad;
    
    /* apply the relu */
    std::vector<float> result_data(this_data.size());
    for (size_t i = 0; i < this_data.size(); i++) {
        result_data[i] = this_data[i] > 0 ? this_data[i] : 0;
    }

    /* construct result tensor */
    Tensor result = Tensor(result_data, this_shape, result_requires_grad);

    /* construt backward function */
    if (result_requires_grad) {
        
        std::thread::id tid = std::this_thread::get_id();

        /* add result to computation graph */
        {
            std::lock_guard<std::mutex> lock(TensorData::GLOBAL_PARENTS_MUTEX);
            if (this_requires_grad) {
                result.ptr->parents[tid].insert(this->ptr);
            }
        }

        /* ensure thread-local gradients are initialized */
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

        /* construct backward function */
        result.ptr->backward_fn = [this_ptr = this->ptr, result_ptr = result.ptr]() {

            std::thread::id tid = std::this_thread::get_id();

            auto this_data = this_ptr->data;
            auto this_grad = this_ptr->thread_gradients[tid];
            auto result_grad = result_ptr->thread_gradients[tid]->data;

            if (this_ptr->requires_grad && this_grad) {
                for (size_t i = 0; i < this_data.size(); i++) {
                    this_grad->data[i] += result_grad[i] * (this_data[i] > 0 ? 1 : 0);
                }
            }
        };
    }

    return result;
}