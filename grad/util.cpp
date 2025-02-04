#include "cppgrad.h"

/**
 * @brief Global atomic counter for assigning unique tensor IDs.
 *
 * Each new tensor gets a unique ID to track dependencies in the computation graph.
 */
std::atomic<std::uint64_t> id_counter = 1;

/**
 * @brief Global random number generator used for weight initialization.
 *
 * The generator is initialized with a fixed seed (`42`) for reproducibility.
 */
std::mt19937 global_generator(42);

/**
 * @brief Global mutex for synchronizing access to thread-local gradients.
 */
std::mutex TensorData::GLOBAL_GRAD_MUTEX;

/**
 * @brief Global mutex for synchronizing access to parent tensors in the computation graph.
 */
std::mutex TensorData::GLOBAL_PARENTS_MUTEX;


/**
 * @brief Generates a unique ID for a new tensor.
 *
 * @return size_t A unique tensor ID.
 * @throws std::runtime_error If the ID counter overflows.
 */
size_t get_id() {

    size_t counter = id_counter++;
    if (counter == 0) {
        throw std::runtime_error("Tensor ID counter overflow");
    }
    return counter;
}

/**
 * @brief Sets the random seed for the global random number generator.
 *
 * @param seed The seed value for deterministic random number generation.
 */
void set_seed(int seed) {
    global_generator.seed(seed);
}

/**
 * @brief Resets the gradient of the tensor to zero.
 *
 * This function is used to clear accumulated gradients before a new backward pass.
 */
void Tensor::zero_grad() {
    std::thread::id tid = std::this_thread::get_id();
    if (ptr->thread_gradients[tid]) {
        std::fill(ptr->thread_gradients[tid]->data.begin(), ptr->thread_gradients[tid]->data.end(), 0.0f);
    }
}

/**
 * @brief Computes the total number of elements in a given shape.
 *
 * @param shp The shape vector representing tensor dimensions.
 * @return size_t The total number of elements.
 */
size_t numel(const std::vector<size_t>& shp) {
    size_t product = 1;
    for (auto s : shp) {
        product *= s;
    }
    return product;
}

/**
 * @brief Computes the broadcasted shape for two tensors.
 *
 * This function follows **NumPy broadcasting rules**:
 * - Two dimensions are compatible if they are equal or one of them is `1`.
 * - If incompatible, an empty vector is returned.
 *
 * @param shape1 The shape of the first tensor.
 * @param shape2 The shape of the second tensor.
 * @return std::vector<size_t> The broadcasted shape, or an empty vector if broadcasting fails.
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

/**
 * @brief Converts a multi-dimensional index into a single linear index.
 *
 * This function assumes **row-major ordering**.
 *
 * @param multi_index A vector representing a multi-dimensional index.
 * @param shape The shape of the tensor.
 * @return size_t The corresponding linear index.
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

/**
 * @brief Converts a linear index into a multi-dimensional index.
 *
 * @param idx The linear index.
 * @param shape The shape of the tensor.
 * @return std::vector<size_t> The corresponding multi-dimensional index.
 */
std::vector<size_t> unravel_index(size_t idx, const std::vector<size_t>& shape) {
    std::vector<size_t> coords(shape.size());
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
        coords[i] = idx % shape[i];
        idx /= shape[i];
    }
    return coords;
}


/**
 * @brief Reduces a gradient tensor to match the original shape after broadcasting.
 *
 * This function accumulates gradients along broadcasted dimensions to maintain
 * correct gradient propagation.
 *
 * @param grad The gradient tensor after the operation.
 * @param grad_shape The shape of the gradient tensor.
 * @param original_shape The original shape of the tensor before broadcasting.
 * @return std::vector<float> The reduced gradient tensor.
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

/**
 * @brief Prints the shape of a tensor to standard output.
 */
void printShape(const std::vector<size_t>& shape) {
    for (size_t s : shape) {
        std::cout << s << " ";
    }
    std::cout << std::endl;
}

/**
 * @brief Prints two tensor shapes side by side for debugging.
 *
 * @param shape1 The shape of the first tensor.
 * @param shape2 The shape of the second tensor.
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

/**
 * @brief Returns the shape of the tensor.
 *
 * @return std::vector<size_t> The shape vector of the tensor.
 */
std::vector<size_t> Tensor::shape() const {
    if(this->ptr == NULL){
        return {};
    }
    return ptr->shape;
}

/**
 * @brief Returns the raw data of the tensor as a vector.
 *
 * @return std::vector<float> The data of the tensor.
 */
std::vector<float> Tensor::data() const {
    if(this->ptr == NULL){
        return {};
    }
    return ptr->data;
};

/**
 * @brief Retrieves the gradient of the tensor.
 *
 * @return Tensor A tensor containing the gradient.
 * @throws std::runtime_error If the tensor has no gradient.
 */
Tensor Tensor::grad() const{
    std::thread::id tid = std::this_thread::get_id();
    if (!ptr->thread_gradients[tid]) {
        throw std::runtime_error("Tensor has no gradient.");
    }
    return Tensor(ptr->thread_gradients[tid]);
}

/**
 * @brief Disables gradient computation for the tensor.
 *
 * This is typically used in inference mode to improve efficiency.
 */
void Tensor::eval() {
    ptr->requires_grad = false;
}

/**
 * @brief Enables gradient computation for the tensor.
 *
 * This is required when training a model.
 */
void Tensor::train() {
    ptr->requires_grad = true;
}