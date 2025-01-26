#include "cppgrad.h"

Tensor Tensor::exp() const{
    // 1) Compute exp elementwise
    std::vector<float> forward_data(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        forward_data[i] = std::exp(data[i]);
    }

    // 2) Create the output tensor
    std::shared_ptr<Tensor> result = std::make_shared<Tensor>(forward_data, shape, requires_grad);
    
    // 3) If we need gradients, set up the backward function
    if (result->requires_grad) {
        // Capture only what's needed in the lambda
        auto this_requires_grad = this->requires_grad;
        auto this_grad    = this->grad;   // Gradient buffer of the input
        auto this_data    = this->data;   // For completeness, if needed
        auto out_data     = result->data;  // The exp(...) values we just computed
        auto result_grad  = result->grad;
        auto this_backward_fn = this->backward_fn;

        if (this_requires_grad)
            result->parents.push_back(std::make_shared<Tensor>(*this));


        result->backward_fn = [=]() mutable {
            if (this_requires_grad && this_grad) {
                for (size_t i = 0; i < this_data.size(); i++) {
                    // out_data[i] is exp(this_data[i])
                    this_grad->data[i] += result_grad->data[i] * out_data[i];
                }
            }
        };
    }

    return *result;
}

Tensor Tensor::log() const {
    // 1) Compute exp elementwise
    std::vector<float> forward_data(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        forward_data[i] = std::log(data[i]);
    }

    // 2) Create the output tensor
    std::shared_ptr<Tensor> result = std::make_shared<Tensor>(forward_data, shape, requires_grad);

    // 3) If we need gradients, set up the backward function
    if (result->requires_grad) {
        // Capture only what's needed in the lambda
        auto this_requires_grad = this->requires_grad;
        auto this_grad    = this->grad;   // Gradient buffer of the input
        auto this_data    = this->data;   // For completeness, if needed
        auto out_data     = result->data;  // The log(...) values we just computed
        auto result_grad  = result->grad;
        auto this_backward_fn = this->backward_fn;

        if (this_requires_grad)
            result->parents.push_back(std::make_shared<Tensor>(*this));

        result->backward_fn = [=]() mutable {
            if (this_requires_grad && this_grad) {
                for (size_t i = 0; i < this_data.size(); i++) {
                    this_grad->data[i] += result_grad->data[i] / this_data[i];
                }
            }
        };
    }

    return *result;
}


Tensor Tensor::softmax(size_t dim) const {
    Tensor exp_tensor = this->exp(); 
    Tensor sum_exp = exp_tensor.sum(dim) + Tensor({1e-6}, {1}); // Add a small epsilon to avoid division by zero
    sum_exp.shape.push_back(1); // Add a new dimension for broadcasting 

    return exp_tensor / sum_exp;
}

Tensor Tensor::softmax() const {
    return softmax(shape.size() - 1);
}

Tensor Tensor::onehot_encode(size_t num_classes) const {
    size_t result_size = this->numel(shape) * num_classes;
    std::vector<float> result_data(result_size, 0.0f);
    
    std::vector<size_t> result_shape = shape;
    result_shape.push_back(num_classes);
    
    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i] >= num_classes) {
            throw std::invalid_argument("Value out of bounds for one-hot encoding");
        }
        std::vector<size_t> multi_index(shape.size(), 0);
        size_t temp = i;
        for (int j = shape.size() - 1; j >= 0; --j) {
            multi_index[j] = temp % shape[j];
            temp /= shape[j];
        }
        multi_index.push_back(data[i]);
        size_t index = map_index(multi_index, result_shape);

        result_data[index] = 1.0f;
    }

    std::shared_ptr<Tensor> result = std::make_shared<Tensor>(result_data, result_shape, requires_grad);

    if (result->requires_grad) {
        auto this_requires_grad = this->requires_grad;
        auto this_grad = this->grad;
        auto result_grad = result->grad;
        auto this_data = this->data;
        auto this_shape = this->shape;
        auto this_backward_fn = this->backward_fn;

        if (this_requires_grad)
            result->parents.push_back(std::make_shared<Tensor>(*this));

        result->backward_fn = [=]() mutable {
            if (this_requires_grad && this_grad) {
                for (size_t i = 0; i < this_data.size(); ++i) {
                    std::vector<size_t> multi_index(shape.size(), 0);
                    size_t temp = i;
                    for (int j = shape.size() - 1; j >= 0; --j) {
                        multi_index[j] = temp % shape[j];
                        temp /= shape[j];
                    }
                    for (size_t j = 0; j < num_classes; ++j) {
                        multi_index.push_back(j);
                        size_t index = map_index(multi_index, result_shape);
                        this_grad->data[i] += result_grad->data[index];
                        multi_index.pop_back();
                    }
                }
            }
        };
    }

    return *result; 

}

Tensor Tensor::relu() const{
    // 1) Compute exp elementwise
    std::vector<float> forward_data(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        forward_data[i] = data[i] > 0 ? data[i] : 0;
    }

    // 2) Create the output tensor
    std::shared_ptr<Tensor> result = std::make_shared<Tensor>(forward_data, shape, requires_grad);

    // 3) If we need gradients, set up the backward function
    if (result->requires_grad) {
        // Capture only what's needed in the lambda
        auto this_requires_grad = this->requires_grad;
        auto this_grad    = this->grad;   // Gradient buffer of the input
        auto this_data    = this->data;   // For completeness, if needed
        auto out_data     = result->data; 
        auto result_grad  = result->grad;
        auto this_backward_fn = this->backward_fn;

        if (this_requires_grad)
            result->parents.push_back(std::make_shared<Tensor>(*this));

        result->backward_fn = [=]() mutable {
            if (this_requires_grad && this_grad) {
                for (size_t i = 0; i < this_data.size(); i++) {
                    this_grad->data[i] += result_grad->data[i] * (this_data[i] > 0 ? 1 : 0);
                }
            }
        };
    }

    return *result;
}

Tensor CrossEntropyLoss(const Tensor& y_pred, const Tensor& y_true) {
    // 1) Compute the softmax of y_pred along the last dimension (class axis)
    Tensor y_pred_softmax = y_pred.softmax(y_pred.shape.size() - 1);  

    // 2) Convert true labels into one-hot representation
    Tensor y_true_one_hot = y_true.onehot_encode(y_pred.shape.back());

    // 3) Compute negative log-likelihood: - sum(one_hot * log(softmax)) along class axis
    Tensor neg_log_likelihood = -(y_true_one_hot * y_pred_softmax.log()).sum(y_pred.shape.size() - 1);

    // 4) Compute mean loss over batch dimension
    Tensor loss = neg_log_likelihood.mean();

    return loss;
}