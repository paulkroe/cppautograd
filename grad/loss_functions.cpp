#include "cppgrad.h"

/**
 * @brief Computes the softmax function along a specified dimension.
 *
 * The softmax function normalizes the input tensor along the given dimension,
 * ensuring that the output values sum to 1 along that axis. It is defined as:
 * \f[
 * \text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
 * \f]
 * 
 * A small constant (\f$10^{-6}\f$) is added to the denominator for numerical stability.
 *
 * @param dim The dimension along which to apply the softmax function.
 * @return Tensor The softmax-normalized tensor.
 *
 * @example
 * @code
 * Tensor logits = Tensor({2.0, 1.0, 0.1}, {3});
 * Tensor probabilities = logits.softmax(0);
 * @endcode
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

/**
 * @brief Computes the softmax function along the last dimension.
 *
 * This function applies the softmax operation along the last axis of the tensor.
 * It is equivalent to calling `softmax(dim)` with `dim = tensor.ndim() - 1`.
 *
 * @return Tensor The softmax-normalized tensor.
 *
 * @example
 * @code
 * Tensor logits = Tensor({{2.0, 1.0, 0.1}, {0.5, 0.7, 0.2}}, {2, 3});
 * Tensor probabilities = logits.softmax();
 * @endcode
 */
Tensor Tensor::softmax() const {
    return softmax(ptr->shape.size() - 1);
}

/**
 * @brief Converts a tensor of class indices into a one-hot encoded representation.
 *
 * This function expands the input tensor along a new trailing dimension, representing
 * class indices as one-hot vectors. The resulting tensor has shape `[..., num_classes]`,
 * where each value is replaced with a one-hot encoded vector.
 *
 * @param num_classes The total number of classes in the encoding.
 * @return Tensor The one-hot encoded tensor.
 * @throws std::invalid_argument If any input value is out of bounds for `num_classes`.
 *
 * @example
 * @code
 * Tensor labels = Tensor({0, 2, 1}, {3});
 * Tensor one_hot = labels.onehot_encode(3);
 * @endcode
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

/**
 * @brief Computes the Cross Entropy Loss between predicted and true labels.
 *
 * This function computes the categorical cross-entropy loss.
 * It first applies softmax to `y_pred` along the last dimension, 
 * then computes the negative log-likelihood weighted by one-hot
 * encoded `y_true`. Finally, it returns the mean loss over the batch dimension.
 *
 * The formula for cross-entropy loss is:
 * \f[
 * L = - \frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{C} y_{ij} \log(\text{softmax}(x_{ij}))
 * \f]
 *
 * @param y_pred Tensor of shape `(batch_size, num_classes)`, representing model logits.
 * @param y_true Tensor of shape `(batch_size)`, containing ground truth class indices.
 * @return Tensor The scalar cross-entropy loss.
 *
 * @example
 * @code
 * Tensor logits = Tensor({{2.0, 1.0, 0.1}, {0.5, 0.7, 0.2}}, {2, 3});
 * Tensor labels = Tensor({0, 2}, {2});
 * Tensor loss = CrossEntropyLoss(logits, labels);
 * @endcode
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