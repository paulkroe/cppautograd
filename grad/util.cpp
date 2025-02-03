#include "cppgrad.h"
std::atomic<std::uint64_t> id_counter = 1;
std::mt19937 global_generator(42);

std::mutex TensorData::GLOBAL_GRAD_MUTEX;
std::mutex TensorData::GLOBAL_PARENTS_MUTEX;

/* helper function to get tensor id*/
size_t get_id() {
    size_t counter = id_counter++;
    if (counter == 0) {
        throw std::runtime_error("Tensor ID counter overflow");
    }
    return counter;
}

/* helper function to set random seed */
void set_seed(int seed) {
    global_generator.seed(seed);
}

/* helper function zeroing out the gradient */
void Tensor::zero_grad() {
    std::thread::id tid = std::this_thread::get_id();
    if (ptr->thread_gradients[tid]) {
        std::fill(ptr->thread_gradients[tid]->data.begin(), ptr->thread_gradients[tid]->data.end(), 0.0f);
    }
}

/* helper function to count the number of elements */
size_t numel(const std::vector<size_t>& shp) {
    size_t product = 1;
    for (auto s : shp) {
        product *= s;
    }
    return product;
}

/* helper function to print the shape of a tensor */
void Tensor::print_shape() const{
    for (size_t s : ptr->shape) {
        std::cout << s << " ";
    }
    std::cout << std::endl;
}

/* 
 * given two shapes, returns the shape that results 
 * from broadcasting the two shapes
 * returns an empty vector if the shapes are not broadcastable
 * when iterating over the dimension sizes, starting at the trailing dimension,
 * the dimension sizes must either be equal, one of them is 1, or one of them does not exist
 */
std::vector<size_t> broadcast(const std::vector<size_t>& shape1, const std::vector<size_t>& shape2) {
    /* number of dimensions in the broadcasted tensor */
    size_t max_dims = std::max(shape1.size(), shape2.size());
    
    /* resulting shape */
    std::vector<size_t> result(max_dims, 1);

    for (size_t i = 0; i < max_dims; ++i) {
        /* get the dimension sizes */
        size_t dim1 = i < shape1.size() ? shape1[shape1.size() - 1 - i] : 1;
        size_t dim2 = i < shape2.size() ? shape2[shape2.size() - 1 - i] : 1;

        /* if unable to broadcast, return empty vector */
        if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
            return {};
        }

        /* set the dimension size in the resulting shape */
        result[max_dims - 1 - i] = std::max(dim1, dim2);
    }

    return result;
}

/*
 * given a multi-dimensional coordinate and a shape, flatten 
 * back to a single offset in row-major order
 */
size_t ravel_index(const std::vector<size_t>& multi_index,
                        const std::vector<size_t>& shape) {
    size_t linear_index = 0;
    size_t stride = 1;

    /* align dimensions: shape may be smaller than multi_index. */
    int offset = multi_index.size() - shape.size();

    // Iterate over shape dimensions from right to left
    for (int i = shape.size() - 1; i >= 0; --i) {
        /* adjust for broadcasting */
        int multi_idx_pos = i + offset;

        /* corresponding index in result shape */
        size_t broadcast_dim_index = multi_index[multi_idx_pos];
        size_t dim_size = shape[i];

        /* if broadcasting occurs along this dimension, always use index 0 */
        size_t index_component = (dim_size == 1) ? 0 : broadcast_dim_index;

        /* accumulate to compute the linear index */
        linear_index += index_component * stride;
        stride *= dim_size;
    }

    return linear_index;
}

/*
 * given a linear index `idx` and a shape vector, produce
 * the multi-dimensional coordinate
 * e.g. shape=[2,3,4], idx in [0..23]
 * => coord=[b,c,d]
 */
std::vector<size_t> unravel_index(size_t idx, const std::vector<size_t>& shape) {
    std::vector<size_t> coords(shape.size());
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
        coords[i] = idx % shape[i];
        idx /= shape[i];
    }
    return coords;
}


/*
 * helper function to compute the gradient reduction for broadcasting
 * Given the gradient of `result`, we adjust it to match the original shape
 * of the input tensors by summing over the broadcasted dimensions.
 */
std::vector<float> reduce_grad(const std::vector<float>& grad, 
                               const std::vector<size_t>& grad_shape, 
                               const std::vector<size_t>& original_shape) {
    std::vector<float> reduced_grad(numel(original_shape), 0.0f);

    for (size_t i = 0; i < grad.size(); ++i) {
        std::vector<size_t> multi_index = unravel_index(i, grad_shape);
        size_t reduced_index = ravel_index(multi_index, original_shape);
        reduced_grad[reduced_index] += grad[i];
    }

    return reduced_grad;
}

/* 
 * helper function to print tensor shape 
 */
void printShape(const std::vector<size_t>& shape) {
    for (size_t s : shape) {
        std::cout << s << " ";
    }
    std::cout << std::endl;
}

/*
 * helper function to print tensor shapes
 */
void printShapes(const std::vector<size_t>& shape1, const std::vector<size_t>& shape2) {
    std::cout << "this->shape ";
    for (auto s : shape1) {
        std::cout << s << " ";
    }
    std::cout << " vs. other.shape ";
    for (auto s : shape2) {
        std::cout << s << " ";
    }
    std::cout << std::endl;
}

/*
 * helper function returning the shape of a tensor 
 */
std::vector<size_t> Tensor::shape() const {
    if(this->ptr == NULL){
        return {};
    }
    return ptr->shape;
}

/*
 * helper function returning the data of a tensor 
 */
std::vector<float> Tensor::data() const {
    if(this->ptr == NULL){
        return {};
    }
    return ptr->data;
};

/*
 * helper function returning the gradient tensor
 */
Tensor Tensor::grad() const{
    std::thread::id tid = std::this_thread::get_id();
    if (!ptr->thread_gradients[tid]) {
        throw std::runtime_error("Tensor has no gradient.");
    }
    return Tensor(ptr->thread_gradients[tid]);
}

/* 
 * disable gradient computation
 */
void Tensor::eval() {
    ptr->requires_grad = false;
}
/*
 * enable gradient computation
 */
void Tensor::train() {
    ptr->requires_grad = true;
}