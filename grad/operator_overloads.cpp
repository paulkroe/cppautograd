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
    
    auto this_shape = this->ptr->shape;
    auto other_shape = other.ptr->shape;

    auto this_data = this->ptr->data;
    auto other_data = other.ptr->data;

    auto this_requires_grad = this->ptr->requires_grad;
    auto other_requires_grad = other.ptr->requires_grad;
    auto result_requires_grad = this_requires_grad || other_requires_grad;


    /* infer result shape */
    std::vector<size_t> result_shape = broadcast(this_shape, other_shape);
    if (result_shape.empty()) {
        printShapes(this_shape, other_shape);
        throw std::invalid_argument("Tensor shapes must be broadcastable for addition.");
    }

    /* allocate memory for result data */
    size_t result_size = numel(result_shape);
    std::vector<float> result_data(result_size);

    /* iterate over result data and perform addition */
    for (size_t i = 0; i < result_size; ++i) {
        std::vector<size_t> multi_index = unravel_index(i, result_shape);
        size_t index_a = ravel_index(multi_index, this_shape);
        size_t index_b = ravel_index(multi_index, other_shape);
        result_data[i] = (this_data[index_a]) + other_data[index_b];
    }

    /* allocate result tensor */
    Tensor result = Tensor(result_data, result_shape, result_requires_grad);

    /* construct backward function */
    if (result_requires_grad) {

        std::thread::id tid = std::this_thread::get_id();

        /* add result to computation graph */
        {
            std::lock_guard<std::mutex> lock(TensorData::GLOBAL_PARENTS_MUTEX);
            if (this_requires_grad) {
                result.ptr->parents[tid].insert(this->ptr);
            }
        }
        {
            std::lock_guard<std::mutex> lock(TensorData::GLOBAL_PARENTS_MUTEX);
            if (other_requires_grad) {
                result.ptr->parents[tid].insert(other.ptr);
            }
        }

        /* Ensure thread-local gradients are initialized */
        std::shared_ptr<TensorData> this_grad, other_grad;
        {
            std::lock_guard<std::mutex> lock(TensorData::GLOBAL_GRAD_MUTEX);
            if (this->ptr->requires_grad) {
                if (!this->ptr->thread_gradients[tid]) {
                    this->ptr->thread_gradients[tid] = std::make_shared<TensorData>(std::vector<float>(this->ptr->data.size(), 0.0f), this_shape, false);
                }
                this_grad = this->ptr->thread_gradients[tid];
            }
        }
        {
            std::lock_guard<std::mutex> lock(TensorData::GLOBAL_GRAD_MUTEX);
            if (other.ptr->requires_grad) {
                if (!other.ptr->thread_gradients[tid]) {
                    other.ptr->thread_gradients[tid] = std::make_shared<TensorData>(std::vector<float>(other.ptr->data.size(), 0.0f), other_shape, false);
                }
                other_grad = other.ptr->thread_gradients[tid];
            }
        }

        /* Capture shared pointers to gradients in backward function */
        result.ptr->backward_fn = [this_ptr = this->ptr, other_ptr = other.ptr, result_ptr = result.ptr]() 
        {
            std::thread::id tid = std::this_thread::get_id();
            std::cout << "Backward function for addition\n";

            /* 1) Gradient w.r.t. x (this->data) */
            auto this_grad = this_ptr->thread_gradients[tid];
            if (this_ptr->requires_grad && this_grad) {
                auto reduced_grad = reduce_grad(result_ptr->thread_gradients[tid]->data, result_ptr->shape, this_ptr->shape);
                for (size_t i = 0; i < reduced_grad.size(); ++i) {
                    this_grad->data[i] += reduced_grad[i];
                }
            }

            /* 2) Gradient w.r.t. y (other->data) */
            auto other_grad = other_ptr->thread_gradients[tid];
            if (other_ptr->requires_grad && other_grad) {
                auto reduced_grad = reduce_grad(result_ptr->thread_gradients[tid]->data, result_ptr->shape, other_ptr->shape);
                for (size_t i = 0; i < reduced_grad.size(); ++i) {
                    other_grad->data[i] += reduced_grad[i];
                }
            }
        };
    }

    return result;
}

/*
 * Overloaded + operator, for adding scalar and tensor
 */
Tensor Tensor::operator+(const float other) const{
    return *this + Tensor({other}, false);
}

// /* 
//  * Overloaded unary - operator
//  * 1. Negate the data
//  * 2. Negate the gradient if necessary
//  * 3. Construct the result tensor
//  * 4. If necessary, set up the backward function
//  */
// Tensor Tensor::operator-() const{
    
//     /* negate data */
//     std::vector<float> result_data(data.size());
//     for (size_t i = 0; i < data.size(); i++) {
//         result_data[i] = -data[i];
//     }
    
//     /* construct result tensor */
//     std::shared_ptr<Tensor> result = std::make_shared<Tensor>(result_data, shape, requires_grad);

//     /* construct backward function */
//     if (result->requires_grad) {
        
//         /*
//          * copy data necessary for backward function
//          * to avoid dangling references
//          */
//         auto sz = data.size();
//         auto this_requires_grad = this->requires_grad;
//         /* Store parent in a thread-safe manner */
//         std::thread::id tid = std::this_thread::get_id();
//         {
//             std::lock_guard<std::mutex> lock(GLOBAL_PARENTS_MUTEX);
//             if (this_requires_grad) result->parents[tid].insert(std::make_shared<Tensor>(*this));
//         }
//         /* Ensure thread-local gradients are initialized */
//         std::shared_ptr<Tensor> this_grad;
//         {
//             std::lock_guard<std::mutex> lock(GLOBAL_GRAD_MUTEX);
//             if (this_requires_grad) {
//                 if (!this->thread_gradients[tid]) {
//                     this->thread_gradients[tid] = std::make_shared<Tensor>(std::vector<float>(this->data.size(), 0.0f), this->shape, false);
//                 }
//                 this_grad = this->thread_gradients[tid];
//             }
//         }

//         result->backward_fn = [sz,
//                               this_requires_grad, this_grad,
//                               result]() {
//             std::thread::id tid = std::this_thread::get_id();

//             if (this_requires_grad && this_grad) {

//                 for (size_t i = 0; i < sz; i++) {
//                     this_grad->data[i] -= result->thread_gradients[tid]->data[i];
//                 }
//             }
//         };
//     }

//     return *result;
// }

// /* 
//  * Overloaded binary - operator
//  * using unary - and + operators
//  * this - other = this + (-other)
//  */
// Tensor Tensor::operator-(const Tensor& other) const{
//     /* a - b =  a + (-b) */
//     return *this + (-other);
// }
// /*
//  * Overloaded - operator, for subtracting scalar and tensor
//  */
// Tensor Tensor::operator-(const float other) const{
//     return *this - Tensor({other}, false);
// }

/* 
 * Overloaded * operator, does support broadcasting
 * 1. Check if tensors are broadcastable
 * 2. iterate over result data
 * 3. construct multi-dimensional index
 * 4. perform multiplication
 * 4. If necessary, set up the backward function
 */
Tensor Tensor::operator*(const Tensor& other) const {
    
    auto this_shape = this->ptr->shape;
    auto other_shape = other.ptr->shape;

    auto this_data = this->ptr->data;
    auto other_data = other.ptr->data;

    auto this_requires_grad = this->ptr->requires_grad;
    auto other_requires_grad = other.ptr->requires_grad;
    auto result_requires_grad = this_requires_grad || other_requires_grad;

    /* infer result shape */
    std::vector<size_t> result_shape;
    result_shape = broadcast(this_shape, other_shape);
    
    if (result_shape.empty()) {
        printShapes(this_shape, other_shape);
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
        size_t index_a = ravel_index(multi_index, this_shape);
        size_t index_b = ravel_index(multi_index, other_shape);

        /* perform addition */
        result_data[i] = this_data[index_a] * other_data[index_b];
    }

    /* allocate result tensor */
    Tensor result = Tensor(result_data, result_shape, result_requires_grad);

    /* construct backward function */
    if (result_requires_grad) {
        
        std::thread::id tid = std::this_thread::get_id();
        
        /* add result to computation graph */
        {
            std::lock_guard<std::mutex> lock(TensorData::GLOBAL_PARENTS_MUTEX);
            if (this_requires_grad) {
                result.ptr->parents[tid].insert(this->ptr);
            }
        }
        {
            std::lock_guard<std::mutex> lock(TensorData::GLOBAL_PARENTS_MUTEX);
            if (other_requires_grad) {
                result.ptr->parents[tid].insert(other.ptr);
            }
        }
        
        /* Ensure thread-local gradients are initialized */
        std::shared_ptr<TensorData> this_grad, other_grad;
        {
            std::lock_guard<std::mutex> lock(TensorData::GLOBAL_GRAD_MUTEX);
            if (this_requires_grad) {
                if (!this->ptr->thread_gradients[tid]) {
                    this->ptr->thread_gradients[tid] = std::make_shared<TensorData>(std::vector<float>(this->ptr->data.size(), 0.0f), this_shape, false);
                }
                this_grad = this->ptr->thread_gradients[tid];
            }
        }
        {
            std::lock_guard<std::mutex> lock(TensorData::GLOBAL_GRAD_MUTEX);
            if (other_requires_grad) {
                if (!other.ptr->thread_gradients[tid]) {
                    other.ptr->thread_gradients[tid] = std::make_shared<TensorData>(std::vector<float>(other.ptr->data.size(), 0.0f), other_shape, false);
                }
                other_grad = other.ptr->thread_gradients[tid];
            }
        }

        result.ptr->backward_fn = [this_ptr = this->ptr, other_ptr = other.ptr, result_ptr = result.ptr]() 
        {
            std::thread::id tid = std::this_thread::get_id();

            auto this_shape = this_ptr->shape;
            auto other_shape = other_ptr->shape;

            auto this_data = this_ptr->data;
            auto other_data = other_ptr->data;

            auto result_data = result_ptr->data;
            auto result_grad = result_ptr->thread_gradients[tid]->data;

            /* 1) Gradient w.r.t. x (this->data) */
            auto this_grad = this_ptr->thread_gradients[tid];
            if (this_ptr->requires_grad && this_grad) {

                /* temporary helper varibale to element wise gradient */
                std::vector<float> partial_grad_x(result_grad.size(), 0.0f);

                /* iterate over result data and compute gradient */
                for (size_t i = 0; i < result_grad.size(); ++i) {
                    /* compute the multi-dimensional index in the result shape */
                    std::vector<size_t> multi_index = unravel_index(i, result_ptr->shape);
                    /* map index into other->data */
                    size_t index_b = ravel_index(multi_index, other_shape);
                    /* compute the gradient */
                    partial_grad_x[i] = result_grad[i] * other_data[index_b];
                }

                /* reduce the gradient by summing over broadcasted dimension */
                auto reduced_grad_x = reduce_grad(partial_grad_x, result_ptr->shape, this_shape);
                
                for (size_t i = 0; i < reduced_grad_x.size(); ++i) {
                    this_grad->data[i] += reduced_grad_x[i];
                }

            } 

            /* 2) Gradient w.r.t. y (other->data) */
            auto other_grad = other_ptr->thread_gradients[tid];
            if (other_ptr->requires_grad && other_grad) {
                /* temporary helper varibale to element wise gradient */
                std::vector<float> partial_grad_y(result_grad.size(), 0.0f);

                /* iterate over result data and compute gradient */
                for (size_t i = 0; i < result_grad.size(); i++) {
                    /* compute the multi-dimensional index in the result shape */
                    std::vector<size_t> multi_index = unravel_index(i, result_ptr->shape);
                    /* map index into this->data */
                    size_t index_a = ravel_index(multi_index, this_shape);
                    /* compute the gradient */
                    partial_grad_y[i] = result_grad[i] * this_data[index_a];
                }
                
                /* reduce the gradient by summing over broadcasted dimension */
                auto reduced_grad_y = reduce_grad(partial_grad_y, result_ptr->shape, other_shape);
                
                for (size_t i = 0; i < reduced_grad_y.size(); ++i) {
                    other_grad->data[i] += reduced_grad_y[i];
                }
            }
        };
    }

    return result;
}

/*
 * Overloaded * operator, for multiplying scalar and tensor
 */
Tensor Tensor::operator*(const float other) const {
    return *this * Tensor({other}, false);
}

// /* 
//  * Overloaded / operator, does support broadcasting
//  * 1. Check if tensors are broadcastable
//  * 2. iterate over result data
//  * 3. construct multi-dimensional index
//  * 4. perform division
//  * 4. If necessary, set up the backward function
//  */
// Tensor Tensor::operator/(const Tensor& other) const {
    
//     /* infer result shape */
//     std::vector<size_t> result_shape;
//     result_shape = broadcast(this->shape, other.shape);
    
//     if (result_shape.empty()) {
//         printShapes(this->shape, other.shape);
//         throw std::invalid_argument("Tensor shapes must be broadcastable for division.");
//     }
            
//     /* allocate memory for result data */
//     size_t result_size = numel(result_shape);
//     std::vector<float> result_data(result_size);

//     /* iterate over result data and perform division */
//     for (size_t i = 0; i < result_size; ++i) {

//         /* compute the multi-dimensional index in the result shape */
//         std::vector<size_t> multi_index = unravel_index(i, result_shape);

//         /* map to indices in the original tensors */
//         size_t index_a = ravel_index(multi_index, this->shape);
//         size_t index_b = ravel_index(multi_index, other.shape);

//         /* Perform the division */
//         result_data[i] = this->data[index_a] / other.data[index_b];
//     }

//     /* allocate result tensor */
//     std::shared_ptr<Tensor> result = std::make_shared<Tensor>(result_data, result_shape, requires_grad || other.requires_grad);

//     /* construct backward function */
//     if (result->requires_grad) {
//         /*
//          * copy data necessary for backward function
//          * to avoid dangling references
//          */
//         auto this_requires_grad = this->requires_grad;
//         auto other_requires_grad = other.requires_grad;
//         auto this_backward_fn = this->backward_fn;
//         auto other_backward_fn = other.backward_fn;

//         std::thread::id tid = std::this_thread::get_id();

//         /* add result to computation graph */
//         {
//             std::lock_guard<std::mutex> lock(GLOBAL_PARENTS_MUTEX);
//             if (this_requires_grad) {
//                 auto parent = std::make_shared<Tensor>(*this);
//                 /* should be the same tensor in the computation graph */
//                 parent->id = this->id;   
//                 result->parents[tid].insert(parent);
//             }
//         }
//         {
//             std::lock_guard<std::mutex> lock(GLOBAL_PARENTS_MUTEX);
//             if (other_requires_grad) {
//                 auto parent = std::make_shared<Tensor>(other);
//                 /* should be the same tensor in the computation graph */
//                 parent->id = other.id;
//                 result->parents[tid].insert(parent);
//             }
//         }

//         /* Ensure thread-local gradients are initialized */
//         std::shared_ptr<Tensor> this_grad, other_grad;
//         {
//             std::lock_guard<std::mutex> lock(GLOBAL_GRAD_MUTEX);
//             if (this_requires_grad) {
//                 if (!this->thread_gradients[tid]) {
//                     this->thread_gradients[tid] = std::make_shared<Tensor>(std::vector<float>(this->data.size(), 0.0f), this->shape, false);
//                 }
//                 this_grad = this->thread_gradients[tid];
//             }
//         }
//         {
//             std::lock_guard<std::mutex> lock(GLOBAL_GRAD_MUTEX);
//             if (other_requires_grad) {
//                 if (!other.thread_gradients[tid]) {
//                     other.thread_gradients[tid] = std::make_shared<Tensor>(std::vector<float>(other.data.size(), 0.0f), other.shape, false);
//                 }
//                 other_grad = other.thread_gradients[tid];
//             }
//         }

//         result->backward_fn = [this_requires_grad, other_requires_grad,
//                     this_grad, other_grad,
//                     this_backward_fn, other_backward_fn,
//                     result, 
//                     this_data = this->data,
//                     other_data = other.data,
//                     this_shape = this->shape,
//                     other_shape = other.shape,
//                     result_shape = result->shape]() 
//         {

//             std::thread::id tid = std::this_thread::get_id();

//             /* 1) Gradient w.r.t. x (this->data) */
//             if (this_requires_grad && this_grad)
//             {
//                 /* temporary helper varibale to element wise gradient */
//                 std::vector<float> partial_grad_x(result->thread_gradients[tid]->data.size(), 0.0f);

//                 /* iterate over result data and compute gradient */
//                 for (size_t i = 0; i < result->thread_gradients[tid]->data.size(); ++i) {
//                     /* compute the multi-dimensional index in the result shape */
//                     std::vector<size_t> multi_index = unravel_index(i, result_shape); 
//                     /* map index into other->data */
//                     size_t index_b = ravel_index(multi_index, other_shape);
//                     /* compute the gradient */
//                     partial_grad_x[i] = result->thread_gradients[tid]->data[i] / other_data[index_b];
//                 }

//                 /* reduce the gradient by summing over broadcasted dimension */
//                 auto reduced_grad_x = reduce_grad(partial_grad_x, result_shape, this_shape);
                
//                 for (size_t i = 0; i < reduced_grad_x.size(); ++i) {
//                     this_grad->data[i] += reduced_grad_x[i];
//                 }
//             } 

//             /* 2) Gradient w.r.t. y (other->data) */
//             if (other_requires_grad && other_grad) {
//                 /* temporary helper varibale to element wise gradient */
//                 std::vector<float> partial_grad_y(result->thread_gradients[tid]->data.size(), 0.0f);

//                 /* iterate over result data and compute gradient */
//                 for (size_t i = 0; i < result->thread_gradients[tid]->data.size(); i++) {
//                     /* compute the multi-dimensional index in the result shape */
//                     std::vector<size_t> multi_index = unravel_index(i, result_shape);
//                     /* map index into data */
//                     size_t index_a = ravel_index(multi_index, this_shape);
//                     size_t index_b = ravel_index(multi_index, other_shape);
//                     /* compute the gradient */
//                     partial_grad_y[i] = -result->thread_gradients[tid]->data[i] * this_data[index_a] / (other_data[index_b] * other_data[index_b]);
//                 }

//                 /* reduce the gradient by summing over broadcasted dimension */
//                 auto reduced_grad_y = reduce_grad(partial_grad_y, result_shape, other_shape);
                
//                 for (size_t i = 0; i < reduced_grad_y.size(); ++i) {
//                     other_grad->data[i] += reduced_grad_y[i];
//                 }
//             }
//         };
//     }

//     return *result;
// }

// /*
//  * Overloaded * operator, for multiplying scalar and tensor
//  */
// Tensor Tensor::operator/(const float other) const {
//     return *this / Tensor({other}, false);
// }

/* 
 * Helper function to the << operator
 */
void Tensor::print_recursive(std::ostream& os, size_t dim, size_t offset, size_t stride) const {
    os << "[";
    for (size_t i = 0; i < ptr->shape[dim]; ++i) {
        if (dim == ptr->shape.size() - 1) { 
            /* last dimension, print values directly */
            os << ptr->data[offset + i];
        } else {
            /* recursively print nested dimensions */
            print_recursive(os, dim + 1, offset + i * stride, stride / ptr->shape[dim + 1]);
        }
        if (i + 1 < ptr->shape[dim]) os << ", ";
    }
    os << "]";
}

/*
 * Overloaded << operator
 * Print tensor data and shape, similar to PyTorch
 */
std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    os << "Tensor(";
    tensor.print_recursive(os, 0, 0, tensor.ptr->data.size() / tensor.ptr->shape[0]);
    os << ", shape=";
    os << "[";
    for (size_t i = 0; i < tensor.ptr->shape.size(); ++i) {
        os << tensor.ptr->shape[i];
        if (i + 1 < tensor.ptr->shape.size()) os << ", ";
    }
    os << "]";
    os << ")";
    return os;
}