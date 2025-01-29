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
        auto this_grad    = this->grad;
        auto this_data    = this->data;
        auto result_grad  = result->grad;

        /* add result to computation graph */
        if (this_requires_grad)
            result->parents.push_back(std::make_shared<Tensor>(*this));

        result->backward_fn = [
            this_requires_grad, this_grad,
            this_data, result_grad
        ](const size_t num_threads) {

            /* serial backward function, num_threads not used */
            (void)num_threads;

            if (this_requires_grad && this_grad) {
                for (size_t i = 0; i < this_data.size(); i++) {
                    this_grad->data[i] += result_grad->data[i] * (this_data[i] > 0 ? 1 : 0);
                }
            }
        };
    }

    return *result;
}