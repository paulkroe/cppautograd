#include "cppgrad.h"
int id_counter = 0;
int get_id() {
    return id_counter++;
}

std::mt19937 global_generator(42);

void set_seed(int seed) {
    global_generator.seed(seed);
}


size_t map_index(const std::vector<size_t>& multi_index,
                        const std::vector<size_t>& shape)
    {
        size_t linear_index = 0;
        size_t stride = 1;

        // Align dimensions: shape may be smaller than multi_index.
        int offset = multi_index.size() - shape.size();

        // Iterate over shape dimensions from right to left
        for (int i = shape.size() - 1; i >= 0; --i) {
            int multi_idx_pos = i + offset; // Adjust for dimension alignment

            size_t broadcast_dim_index = multi_index[multi_idx_pos];  // Corresponding index in result shape
            size_t dim_size = shape[i];  // Size of dimension in original tensor

            // If broadcasting occurs along this dimension, always use index 0
            size_t index_component = (dim_size == 1) ? 0 : broadcast_dim_index;

            // Accumulate to compute the linear index
            linear_index += index_component * stride;
            stride *= dim_size;
        }

        return linear_index;
    }

/* 
 * Helper function to infer broadcast shape 
 * To shapes are broadcastable 
 * When iterating over the dimension sizes, starting at the trailing dimension,
 * the dimension sizes must either be equal, one of them is 1, or one of them does not exist
 */
std::vector<size_t> broadcast(const std::vector<size_t>& shape1, const std::vector<size_t>& shape2) {
    size_t max_dims = std::max(shape1.size(), shape2.size());
    std::vector<size_t> result(max_dims, 1);

    for (size_t i = 0; i < max_dims; ++i) {
        size_t dim1 = i < shape1.size() ? shape1[shape1.size() - 1 - i] : 1;
        size_t dim2 = i < shape2.size() ? shape2[shape2.size() - 1 - i] : 1;

        if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
            return {};
        }

        result[max_dims - 1 - i] = std::max(dim1, dim2);
    }

    return result;
}

/*
 * helper function to compute the gradient reduction for broadcasting
 * Given the gradient of `result`, we adjust it to match the original shape
 * of the input tensors by summing over the broadcasted dimensions.
 */

std::vector<float> Tensor::reduce_grad(const std::vector<float>& grad, 
                               const std::vector<size_t>& grad_shape, 
                               const std::vector<size_t>& original_shape) {
    std::vector<float> reduced_grad(Tensor::numel(original_shape), 0.0f);

    for (size_t i = 0; i < grad.size(); ++i) {
        std::vector<size_t> multi_index = unravel_index(i, grad_shape);
        size_t reduced_index = map_index(multi_index, original_shape);
        reduced_grad[reduced_index] += grad[i];
    }

    return reduced_grad;
}

std::vector<size_t> unravel_index(size_t idx, const std::vector<size_t>& shape) {
    std::vector<size_t> coords(shape.size());
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
        coords[i] = idx % shape[i];
        idx /= shape[i];
    }
    return coords;
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