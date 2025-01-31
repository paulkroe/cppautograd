#include "cppgrad.h"

/* 
 * softmax function along a given dimension
 */
Tensor Tensor::softmax(size_t dim) const {
    /* exponentiate tensor */
    Tensor exp_tensor = this->exp();
    /* sum over exponentiated tensor */ 
    Tensor sum_exp = exp_tensor.sum(dim) + Tensor({1e-6}, {1}); /* Add a small epsilon to avoid division by zero */
    
    /* add trailing dimension for broadcasting */
    sum_exp.shape.insert(sum_exp.shape.begin() + dim, 1);

    return exp_tensor / sum_exp;
}

/* 
 * softmax function along trailing dimension
 */
Tensor Tensor::softmax() const {
    return softmax(shape.size() - 1);
}

/* 
 * one-hot encoding of the tensor
 * num_classes: number of classesz
 */
Tensor Tensor::onehot_encode(size_t num_classes) const {

    /* allocate memory for result data */
    size_t result_size = this->numel(shape) * num_classes;
    std::vector<float> result_data(result_size, 0.0f);
    
    std::vector<size_t> result_shape = shape;
    result_shape.push_back(num_classes);
    
    /* iterate over data and expand in the trailing dimension */
    for (size_t i = 0; i < data.size(); ++i) {

        /* check if value is out of bounds */
        if (data[i] >= num_classes) {
            throw std::invalid_argument("Value out of bounds for one-hot encoding");
        }

        /* construct multi-index into this->shape */
        std::vector<size_t> multi_index = unravel_index(i, shape);
        
        multi_index.push_back(data[i]);
        /* map index into result's shape */
        size_t index = ravel_index(multi_index, result_shape);

        result_data[index] = 1.0f;
    }

    /* allocate result tensor */
    std::shared_ptr<Tensor> result = std::make_shared<Tensor>(result_data, result_shape, requires_grad);

    /* construct backward function */
    if (result->requires_grad) {
        /*
         * copy data necessary for backward function
         * to avoid dangling references
         */
        auto this_requires_grad = this->requires_grad;
        auto this_data = this->data;
        auto this_shape = this->shape;
        auto this_backward_fn = this->backward_fn;

        std::thread::id tid = std::this_thread::get_id();

        /* Store parents in a thread-safe manner */
        {
            std::lock_guard<std::mutex> lock(GLOBAL_PARENTS_MUTEX);
            if (this_requires_grad) result->parents[tid].insert(std::make_shared<Tensor>(*this));
        }
        
        /* Ensure thread-local gradients are initialized */
        std::shared_ptr<Tensor> this_grad;
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
            this_requires_grad, this_grad,
            result, this_data, this_shape,
            result_shape, num_classes
        ]() {

            std::thread::id tid = std::this_thread::get_id();

            for (size_t i = 0; i < this_data.size(); ++i) {
                /* compute the multi-dimensional index in this->shape */
                std::vector<size_t> multi_index = unravel_index(i, this_shape);
                
                /* iterate over classes and propagrate the gradient */
                multi_index.push_back(0);
                for (size_t j = 0; j < num_classes; ++j) {
                    multi_index[this_shape.size() - 1] = j;
                    size_t index = ravel_index(multi_index, result_shape);
                    this_grad->data[i] += result->thread_gradients[tid]->data[index];
                }
            }
        };
    }

    return *result; 

}

/*
 * Cross Entropy Loss
 * y_pred: Tensor of shape (batch_size, num_classes)
 * y_true: Tensor of shape (batch_size)
 */
Tensor CrossEntropyLoss(const Tensor& y_pred, const Tensor& y_true) {
    /* compute the softmax of y_pred along the last dimension (class axis) */
    Tensor y_pred_softmax = y_pred.softmax(y_pred.shape.size() - 1);  

    /* convert true labels into one-hot representation */
    Tensor y_true_one_hot = y_true.onehot_encode(y_pred.shape.back());

    /* compute negative log-likelihood: - sum(one_hot * log(softmax)) along class axis */
    Tensor neg_log_likelihood = -(y_true_one_hot * y_pred_softmax.log()).sum(y_pred.shape.size() - 1);

    /* compute mean loss over batch dimension */
    Tensor loss = neg_log_likelihood.mean();

    return loss;
}