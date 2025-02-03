#include "cppgrad.h"

/* 
 * softmax function along a given dimension
 */
Tensor Tensor::softmax(size_t dim) const {
    /* exponentiate tensor */
    Tensor exp_tensor = this->exp();
    /* sum over exponentiated tensor */ 
    Tensor sum_exp = exp_tensor.sum(dim) + 1e-6; 

    /* add trailing dimension for broadcasting */
    sum_exp.ptr->shape.insert(sum_exp.ptr->shape.begin() + dim, 1);

    return exp_tensor / sum_exp;
}

/* 
 * softmax function along trailing dimension
 */
Tensor Tensor::softmax() const {
    return softmax(ptr->shape.size() - 1);
}

/* 
 * one-hot encoding of the tensor
 * num_classes: number of classesz
 */
Tensor Tensor::onehot_encode(size_t num_classes) const {

    auto this_shape = ptr->shape;
    auto this_data = ptr->data;

    auto this_requires_grad = ptr->requires_grad;

    
    /* allocate memory for result data */
    size_t result_size = numel(this_shape) * num_classes;
    std::vector<float> result_data(result_size, 0.0f);
    
    std::vector<size_t> result_shape = this_shape;
    result_shape.push_back(num_classes);
    
    /* iterate over data and expand in the trailing dimension */
    for (size_t i = 0; i < this_data.size(); ++i) {

        /* check if value is out of bounds */
        if (this_data[i] >= num_classes) {
            throw std::invalid_argument("Value out of bounds for one-hot encoding");
        }

        /* construct multi-index into this->shape */
        std::vector<size_t> multi_index = unravel_index(i, this_shape);
        
        multi_index.push_back(this_data[i]);
        /* map index into result's shape */
        size_t index = ravel_index(multi_index, result_shape);

        result_data[index] = 1.0f;
    }

    /* allocate result tensor */
    Tensor result = Tensor(result_data, result_shape, this_requires_grad);

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
        result.ptr->backward_fn = [this_ptr = this->ptr, result_ptr = result.ptr, num_classes]() {

            std::thread::id tid = std::this_thread::get_id();

            auto this_shape = this_ptr->shape;
            auto result_shape = result_ptr->shape;

            auto this_data = this_ptr->data;
            auto result_data = result_ptr->data;

            auto this_grad = this_ptr->thread_gradients[tid];
            auto result_grad = result_ptr->thread_gradients[tid]->data;

            for (size_t i = 0; i < this_data.size(); ++i) {
                /* compute the multi-dimensional index in this->shape */
                std::vector<size_t> multi_index = unravel_index(i, this_shape);
                
                /* iterate over classes and propagrate the gradient */
                multi_index.push_back(0);
                for (size_t j = 0; j < num_classes; ++j) {
                    multi_index[this_shape.size() - 1] = j;
                    size_t index = ravel_index(multi_index, result_shape);
                    this_grad->data[i] += result_grad[index];
                }
            }
        };
    }

    return result; 

}

/*
 * Cross Entropy Loss
 * y_pred: Tensor of shape (batch_size, num_classes)
 * y_true: Tensor of shape (batch_size)
 */
Tensor CrossEntropyLoss(const Tensor& y_pred, const Tensor& y_true) {
    /* compute the softmax of y_pred along the last dimension (class axis) */
    Tensor y_pred_softmax = y_pred.softmax(y_pred.ptr->shape.size() - 1);  

    /* convert true labels into one-hot representation */
    Tensor y_true_one_hot = y_true.onehot_encode(y_pred.ptr->shape.back());

    /* compute negative log-likelihood: - sum(one_hot * log(softmax)) along class axis */
    Tensor neg_log_likelihood = -(y_true_one_hot * y_pred_softmax.log()).sum(y_pred.ptr->shape.size() - 1);

    /* compute mean loss over batch dimension */
    Tensor loss = neg_log_likelihood.mean();

    return loss;
}