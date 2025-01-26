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
        auto this_grad    = this->grad;
        auto this_data    = this->data;
        auto out_data     = result->data;
        auto result_grad  = result->grad;

        /* add result to computation graph */
        if (this_requires_grad)
            result->parents.push_back(std::make_shared<Tensor>(*this));

        result->backward_fn = [
            this_requires_grad, this_grad,
            this_data, out_data, result_grad
        ]() {
            if (this_requires_grad && this_grad) {
                for (size_t i = 0; i < this_data.size(); i++) {
                    this_grad->data[i] += result_grad->data[i] / this_data[i];
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
        auto this_grad    = this->grad;
        auto this_data    = this->data;
        auto out_data     = result->data;
        auto result_grad  = result->grad;

        /* add result to computation graph */
        if (this_requires_grad)
            result->parents.push_back(std::make_shared<Tensor>(*this));


        result->backward_fn = [=]() mutable {
            if (this_requires_grad && this_grad) {
                for (size_t i = 0; i < this_data.size(); i++) {
                    this_grad->data[i] += result_grad->data[i] * out_data[i];
                }
            }
        };
    }

    return *result;
}