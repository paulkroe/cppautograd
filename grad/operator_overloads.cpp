#include "cppgrad.h"

/* 
 * Overloaded + operator, does support broadcasting
 * 1. Check if the shapes of the tensors match or the tensors are broadcastable
 * 2. Perform elementwise addition
 * 3. Construct the result tensor
 * 4. If necessary, set up the backward function
 */
Tensor Tensor::operator+(const Tensor& other) const{
   
    /* infer result shape */
    std::vector<size_t> result_shape;
    result_shape = broadcast(this->shape, other.shape);
    if (result_shape.empty()) {
        printShapes(this->shape, other.shape);
        throw std::invalid_argument("Tensor shapes must be broadcastable for addition.");
    }
               
    /* allocate memory for result data */
    size_t result_size = numel(result_shape);
    std::vector<float> result_data(result_size);

    /* iterate over result data and perform addition */
    for (size_t i = 0; i < result_size; ++i) {

        /* Compute the multi-dimensional index in the result shape */
        std::vector<size_t> multi_index(result_shape.size(), 0);
        size_t temp = i;
        for (int j = result_shape.size() - 1; j >= 0; --j) {
            multi_index[j] = temp % result_shape[j];
            temp /= result_shape[j];
        }

        /* Map to indices in the original tensors */
        size_t index_a = map_index(multi_index, this->shape);
        size_t index_b = map_index(multi_index, other.shape);

        /* Perform the addition */
        result_data[i] = this->data[index_a] + other.data[index_b];
    }

    /* allocate result tensor */
    std::shared_ptr<Tensor> result = std::make_shared<Tensor>(result_data, result_shape, requires_grad || other.requires_grad);

    /* if required, setup the backward function */
    if (result->requires_grad) {
        
        /*
         * copy data necessary for backward function
         * to avoid dangling references
         */
        auto sz = data.size();
        auto this_requires_grad = this->requires_grad;
        auto other_requires_grad = other.requires_grad;
        auto this_grad = this->grad;
        auto other_grad = other.grad;
        auto this_backward_fn = this->backward_fn;
        auto other_backward_fn = other.backward_fn;
        auto result_grad = result->grad;

        /* insert result into the computation graph */
        if (this_requires_grad)
            result->parents.push_back(std::make_shared<Tensor>(*this));
        if (other_requires_grad)
            result->parents.push_back(std::make_shared<Tensor>(other));

        result->backward_fn = [sz,
                            this_requires_grad, other_requires_grad,
                            this_grad, other_grad,
                            this_backward_fn, other_backward_fn,
                            result_grad, this_shape = this->shape, other_shape = other.shape,
                            result_shape = result->shape]() {

            /*
             * Lambda function to compute the gradient reduction for broadcasting
             * Given the gradient of `result`, we adjust it to match the original shape
             * of the input tensors by summing over the broadcasted dimensions.
             */
            auto reduce_broadcasted_grad = [](
                const std::vector<float>& grad_data,
                const std::vector<size_t>& grad_shape,
                const std::vector<size_t>& original_shape
            ) -> std::vector<float> 
            {
                /* Compute the number of elements in the original tensor */
                size_t original_numel = numel(original_shape);
                
                std::vector<float> reduced_grad(original_numel, 0.0f);

                /* iterate over result grad_data and perform addition */
                for (size_t i = 0; i < grad_data.size(); ++i) {

                    /* Compute the multi-dimensional index in the result shape */
                    std::vector<size_t> multi_index(grad_shape.size(), 0);
                    size_t temp = i;
                    for (int j = grad_shape.size() - 1; j >= 0; --j) {
                        multi_index[j] = temp % grad_shape[j];
                        temp /= grad_shape[j];
                    }

                    /* Map to indices in the original tensor */
                    size_t index = map_index(multi_index, original_shape);

                    /* Accumulate the gradient by summing over broadcasted axes */
                    reduced_grad[index] += grad_data[i];
                }

                return reduced_grad;
            };


            // Backpropagate gradient for the first tensor
            if (this_requires_grad && this_grad) {
                auto reduced_grad = reduce_broadcasted_grad(result_grad->data, result_shape, this_shape);
                for (size_t i = 0; i < reduced_grad.size(); ++i) {
                    this_grad->data[i] += reduced_grad[i];
                }
            }

            // Backpropagate gradient for the second tensor
            if (other_requires_grad && other_grad) {
                auto reduced_grad = reduce_broadcasted_grad(result_grad->data, result_shape, other_shape);
                for (size_t i = 0; i < reduced_grad.size(); ++i) {
                    other_grad->data[i] += reduced_grad[i];
                }
            }
        };

    }

    return *result;

}

/* 
 * Helper function to the << operator
 */
void Tensor::print_recursive(std::ostream& os, size_t dim, size_t offset, size_t stride) const {
    os << "[";
    for (size_t i = 0; i < shape[dim]; ++i) {
        if (dim == shape.size() - 1) { 
            // Last dimension, print values directly
            os << data[offset + i];
        } else {
            // Recursively print nested dimensions
            print_recursive(os, dim + 1, offset + i * stride, stride / shape[dim + 1]);
        }
        if (i + 1 < shape[dim]) os << ", ";
    }
    os << "]";
}

/*
 * Overloaded << operator
 * Print tensor data and shape, similar to PyTorch
 */
std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    os << "Tensor(";
    tensor.print_recursive(os, 0, 0, tensor.data.size() / tensor.shape[0]);
    os << ", shape=";
    os << "[";
    for (size_t i = 0; i < tensor.shape.size(); ++i) {
        os << tensor.shape[i];
        if (i + 1 < tensor.shape.size()) os << ", ";
    }
    os << "]";
    os << ")";
    return os;
}