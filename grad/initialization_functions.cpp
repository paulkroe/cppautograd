#include "cppgrad.h"

/**
 * @brief Creates a tensor with elements sampled from a uniform distribution U(0, 1).
 *
 * This function generates a tensor of the given shape where each element is drawn
 * independently from a uniform distribution in the range \f$[0, 1]\f$.
 *
 * @param shape The desired shape of the tensor.
 * @param requires_grad If `true`, enables gradient computation for this tensor.
 * @return Tensor A tensor initialized with random values sampled from \f$U(0, 1)\f$.
 *
 * @note Uses a global random number generator (`global_generator`).
 *
 * @example
 * @code
 * Tensor t = Tensor::randn({3, 3}, true); // 3x3 random tensor, requires gradient
 * @endcode
 */
Tensor Tensor::randn(const std::vector<size_t>& shape, bool requires_grad) {
    /* allocate memory for tensor */
    size_t total_elems = numel(shape);
    std::vector<float> data(total_elems);

    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    for (size_t i = 0; i < total_elems; i++) {
        data[i] = distribution(global_generator);
    }

    return Tensor(data, shape, requires_grad);
}

/**
 * @brief Creates a tensor using He initialization.
 *
 * He initialization is designed for layers with ReLU activations to improve weight scaling.
 * The elements are sampled from a uniform distribution scaled by a standard deviation
 * of \f$\sqrt{\frac{2}{\text{in_features}}}\f$.
 *
 * @param in_features Number of input features (fan-in).
 * @param out_features Number of output features.
 * @param requires_grad If `true`, enables gradient computation for this tensor.
 * @return Tensor A tensor of shape `(in_features, out_features)` initialized using He initialization.
 *
 * @note Uses a global random number generator (`global_generator`).
 *
 * @example
 * @code
 * Tensor weights = Tensor::randn_he(128, 64, true); // He-initialized 128x64 weight matrix
 * @endcode
 */
Tensor Tensor::randn_he(size_t in_features, size_t out_features, bool requires_grad) {
    /* allocate memory for tensor */
    std::vector<float> data(in_features * out_features);
    
    float stddev = std::sqrt(2.0f / in_features);
    std::uniform_real_distribution<float> distribution(-1.0, 1.0);

    for (size_t i = 0; i < data.size(); i++) {
        data[i] = stddev * distribution(global_generator);
    }

    return Tensor(data, {in_features, out_features}, requires_grad);
}

/**
 * @brief Creates a bias tensor with uniform initialization following PyTorch's convention.
 *
 * The bias values are sampled uniformly from the range \f$[-b, b]\f$, where
 * \f$b = \frac{1}{\sqrt{\text{in_features}}}\f$. This method follows PyTorch's
 * default bias initialization.
 *
 * @param in_features Number of input features (fan-in).
 * @param requires_grad If `true`, enables gradient computation for this tensor.
 * @return Tensor A tensor of shape `(in_features,)` initialized for use as a bias vector.
 *
 * @note Uses a global random number generator (`global_generator`).
 *
 * @example
 * @code
 * Tensor bias = Tensor::bias_uniform(128, true); // Bias tensor for 128 input features
 * @endcode
 */
Tensor Tensor::bias_uniform(size_t in_features, bool requires_grad) {
    /* allocate memory for tensor */
    std::vector<float> data(in_features);

    float bound = 1.0f / std::sqrt(in_features);
    std::uniform_real_distribution<float> distribution(-bound, bound);

    for (size_t i = 0; i < data.size(); i++) {
        data[i] = distribution(global_generator);
    }

    return Tensor(data, {in_features}, requires_grad);
}