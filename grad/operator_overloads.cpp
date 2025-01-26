#include "cppgrad.h"

/* 
 * Overloaded + operator, does support broadcasting
 * 1. check if tensors are broadcastable
 * 2. iterate over result data
 * 3. construct multi-dimensional index
 * 4. perform addition
 * 4. if necessary, set up the backward function
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

        /* compute the multi-dimensional index in the result shape */
        std::vector<size_t> multi_index = unravel_index(i, result_shape);

        /* map to indices in the original tensors */
        size_t index_a = map_index(multi_index, this->shape);
        size_t index_b = map_index(multi_index, other.shape);

        /* perform addition */
        result_data[i] = this->data[index_a] + other.data[index_b];
    }

    /* allocate result tensor */
    std::shared_ptr<Tensor> result = std::make_shared<Tensor>(result_data, result_shape, requires_grad || other.requires_grad);

    /* construct backward function */
    if (result->requires_grad) {
        /*
         * copy data necessary for backward function
         * to avoid dangling references
         */
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

        result->backward_fn = [
                    this_requires_grad, other_requires_grad,
                    this_grad, other_grad,
                    this_backward_fn, other_backward_fn,
                    result_grad, this_shape = this->shape, other_shape = other.shape,
                    result_shape = result->shape]() 
        {

            /* 1) Gradient w.r.t. x (this->data) */
            if (this_requires_grad && this_grad) {
                auto reduced_grad = reduce_grad(result_grad->data, result_shape, this_shape);
                for (size_t i = 0; i < reduced_grad.size(); ++i) {
                    this_grad->data[i] += reduced_grad[i];
                }
            }

            /* 2) Gradient w.r.t. y (other->data) */
            if (other_requires_grad && other_grad) {
                auto reduced_grad = reduce_grad(result_grad->data, result_shape, other_shape);
                for (size_t i = 0; i < reduced_grad.size(); ++i) {
                    other_grad->data[i] += reduced_grad[i];
                }
            }
        };

    }

    return *result;

}

/* 
 * Overloaded unary - operator
 * 1. Negate the data
 * 2. Negate the gradient if necessary
 * 3. Construct the result tensor
 * 4. If necessary, set up the backward function
 */
Tensor Tensor::operator-() const {
    
    /* negate data */
    std::vector<float> result_data(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        result_data[i] = -data[i];
    }
    
    /* construct result tensor */
    std::shared_ptr<Tensor> result = std::make_shared<Tensor>(result_data, shape, requires_grad);

    /* construct backward function */
    if (result->requires_grad) {
        
        /*
         * copy data necessary for backward function
         * to avoid dangling references
         */
        auto sz = data.size();
        auto this_requires_grad = this->requires_grad;
        auto this_grad = this->grad;
        auto this_backward_fn = this->backward_fn;
        auto result_grad = result->grad;

        /* insert result into the computation graph */
        if (this_requires_grad)
            result->parents.push_back(std::make_shared<Tensor>(*this));

        result->backward_fn = [sz,
                              this_requires_grad, this_grad,
                              this_backward_fn, result_grad]() {
            if (this_requires_grad && this_grad) {
               /*
                * If this tensor requires its gradient and
                * the gradient vector exists, update its gradient 
                */
                for (size_t i = 0; i < sz; i++) {
                    this_grad->data[i] -= result_grad->data[i];
                }
            }
        };
    }

    return *result;
}

/* 
 * Overloaded binary - operator
 * using unary - and + operators
 * this - other = this + (-other)
 */
Tensor Tensor::operator-(const Tensor& other) const{
    /* a - b =  a + (-b) */
    return *this + (-other);
}

/* 
 * Overloaded * operator, does support broadcasting
 * 1. Check if tensors are broadcastable
 * 2. iterate over result data
 * 3. construct multi-dimensional index
 * 4. perform multiplication
 * 4. If necessary, set up the backward function
 */
Tensor Tensor::operator*(const Tensor& other) const {
    
    /* infer result shape */
    std::vector<size_t> result_shape;
    result_shape = broadcast(this->shape, other.shape);
    
    if (result_shape.empty()) {
        printShapes(this->shape, other.shape);
        throw std::invalid_argument("Tensor shapes must be broadcastable for multiplication.");
    }
            
    /* allocate memory for result data */
    size_t result_size = numel(result_shape);
    std::vector<float> result_data(result_size);

    /* iterate over result data and perform multiplication */
    for (size_t i = 0; i < result_size; ++i) {
        
        /* compute the multi-dimensional index in the result shape */
        std::vector<size_t> multi_index = unravel_index(i, result_shape);

        /* map to indices in the original tensors */
        size_t index_a = map_index(multi_index, this->shape);
        size_t index_b = map_index(multi_index, other.shape);

        /* perform addition */
        result_data[i] = this->data[index_a] * other.data[index_b];
    }

    /* allocate result tensor */
    std::shared_ptr<Tensor> result = std::make_shared<Tensor>(result_data, result_shape, requires_grad || other.requires_grad);

    /* construct backward function */
    if (result->requires_grad) {
        /*
         * copy data necessary for backward function
         * to avoid dangling references
         */
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
        
        result->backward_fn = [
                    this_requires_grad, other_requires_grad,
                    this_grad, other_grad,
                    this_backward_fn, other_backward_fn,
                    result_grad, 
                    this_data = this->data,
                    other_data = other.data,
                    this_shape = this->shape,
                    other_shape = other.shape,
                    result_shape = result->shape]() 
        {

            /* 1) Gradient w.r.t. x (this->data) */
            if (this_requires_grad && this_grad) {
                /* temporary helper varibale to element wise gradient */
                std::vector<float> partial_grad_x(result_grad->data.size(), 0.0f);

                /* iterate over result data and compute gradient */
                for (size_t i = 0; i < result_grad->data.size(); ++i) {
                    /* compute the multi-dimensional index in the result shape */
                    std::vector<size_t> multi_index = unravel_index(i, result_shape);
                    /* map index into other->data */
                    size_t index_b = map_index(multi_index, other_shape);
                    /* compute the gradient */
                    partial_grad_x[i] = result_grad->data[i] * other_data[index_b];
                }

                /* reduce the gradient by summing over broadcasted dimension */
                auto reduced_grad_x = reduce_grad(partial_grad_x, result_shape, this_shape);
                
                for (size_t i = 0; i < reduced_grad_x.size(); ++i) {
                    this_grad->data[i] += reduced_grad_x[i];
                }

            } 

            /* 2) Gradient w.r.t. y (other->data) */
            if (other_requires_grad && other_grad) {
                /* temporary helper varibale to element wise gradient */
                std::vector<float> partial_grad_y(result_grad->data.size(), 0.0f);

                /* iterate over result data and compute gradient */
                for (size_t i = 0; i < result_grad->data.size(); i++) {
                    /* compute the multi-dimensional index in the result shape */
                    std::vector<size_t> multi_index = unravel_index(i, result_shape);
                    /* map index into this->data */
                    size_t index_a = map_index(multi_index, this_shape);
                    /* compute the gradient */
                    partial_grad_y[i] = result_grad->data[i] * this_data[index_a];
                }
                
                /* reduce the gradient by summing over broadcasted dimension */
                auto reduced_grad_y = reduce_grad(partial_grad_y, result_shape, other_shape);
                
                for (size_t i = 0; i < reduced_grad_y.size(); ++i) {
                    other_grad->data[i] += reduced_grad_y[i];
                }
            }
        };
    }

    return *result;
}

/* 
 * Overloaded / operator, does support broadcasting
 * 1. Check if tensors are broadcastable
 * 2. iterate over result data
 * 3. construct multi-dimensional index
 * 4. perform division
 * 4. If necessary, set up the backward function
 */
Tensor Tensor::operator/(const Tensor& other) const {
    
    /* infer result shape */
    std::vector<size_t> result_shape;
    result_shape = broadcast(this->shape, other.shape);
    
    if (result_shape.empty()) {
        printShapes(this->shape, other.shape);
        throw std::invalid_argument("Tensor shapes must be broadcastable for division.");
    }
            
    /* allocate memory for result data */
    size_t result_size = numel(result_shape);
    std::vector<float> result_data(result_size);

    /* iterate over result data and perform division */
    for (size_t i = 0; i < result_size; ++i) {

        /* compute the multi-dimensional index in the result shape */
        std::vector<size_t> multi_index = unravel_index(i, result_shape);

        /* map to indices in the original tensors */
        size_t index_a = map_index(multi_index, this->shape);
        size_t index_b = map_index(multi_index, other.shape);

        /* Perform the division */
        result_data[i] = this->data[index_a] / other.data[index_b];
    }

    /* allocate result tensor */
    std::shared_ptr<Tensor> result = std::make_shared<Tensor>(result_data, result_shape, requires_grad || other.requires_grad);

    /* construct backward function */
    if (result->requires_grad) {
        /*
         * copy data necessary for backward function
         * to avoid dangling references
         */
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

        result->backward_fn = [this_requires_grad, other_requires_grad,
                    this_grad, other_grad,
                    this_backward_fn, other_backward_fn,
                    result_grad, 
                    this_data = this->data,
                    other_data = other.data,
                    this_shape = this->shape,
                    other_shape = other.shape,
                    result_shape = result->shape]() 
        {
            /* 1) Gradient w.r.t. x (this->data) */
            if (this_requires_grad && this_grad)
            {
                /* temporary helper varibale to element wise gradient */
                std::vector<float> partial_grad_x(result_grad->data.size(), 0.0f);

                /* iterate over result data and compute gradient */
                for (size_t i = 0; i < result_grad->data.size(); ++i) {
                    /* compute the multi-dimensional index in the result shape */
                    std::vector<size_t> multi_index = unravel_index(i, result_shape); 
                    /* map index into other->data */
                    size_t index_b = map_index(multi_index, other_shape);
                    /* compute the gradient */
                    partial_grad_x[i] = result_grad->data[i] / other_data[index_b];
                }

                /* reduce the gradient by summing over broadcasted dimension */
                auto reduced_grad_x = reduce_grad(partial_grad_x, result_shape, this_shape);
                
                for (size_t i = 0; i < reduced_grad_x.size(); ++i) {
                    this_grad->data[i] += reduced_grad_x[i];
                }
            } 

            /* 2) Gradient w.r.t. y (other->data) */
            if (other_requires_grad && other_grad) {
                /* temporary helper varibale to element wise gradient */
                std::vector<float> partial_grad_y(result_grad->data.size(), 0.0f);

                /* iterate over result data and compute gradient */
                for (size_t i = 0; i < result_grad->data.size(); i++) {
                    /* compute the multi-dimensional index in the result shape */
                    std::vector<size_t> multi_index = unravel_index(i, result_shape);
                    /* map index into data */
                    size_t index_a = map_index(multi_index, this_shape);
                    size_t index_b = map_index(multi_index, other_shape);
                    /* compute the gradient */
                    partial_grad_y[i] = -result_grad->data[i] * this_data[index_a] / (other_data[index_b] * other_data[index_b]);
                }

                /* reduce the gradient by summing over broadcasted dimension */
                auto reduced_grad_y = reduce_grad(partial_grad_y, result_shape, other_shape);
                
                for (size_t i = 0; i < reduced_grad_y.size(); ++i) {
                    other_grad->data[i] += reduced_grad_y[i];
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
            /* last dimension, print values directly */
            os << data[offset + i];
        } else {
            /* recursively print nested dimensions */
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