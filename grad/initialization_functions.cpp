#include "cppgrad.h"

/* 
 * initialize a random tensor
 * each element is sampled from U[0, 1]
 */
Tensor Tensor::randn(const std::vector<size_t>& shape, bool requires_grad = false) {
    /* allocate memory for tensor */
    size_t total_elems = Tensor::numel(shape);
    std::vector<float> data(total_elems);

    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    for (size_t i = 0; i < total_elems; i++) {
        data[i] = distribution(global_generator);
    }

    return Tensor(data, shape, requires_grad);
}

/*
 * initialize a random tensor using He initialization
 * sampled from a uniform distribution scaled by stddev.
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

/*
 * initialize a bias tensor with values sampled uniformly from [-bound, bound],
 * where bound = 1 / sqrt(in_features), following PyTorch's bias initialization.
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