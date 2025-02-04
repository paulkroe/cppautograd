#include "./cppgrad.h"

#ifndef MODELS_H
#define MODELS_H

/**
 * @brief Fully connected linear layer.
 *
 * This class implements a standard dense (fully connected) layer commonly used in neural networks.
 * The layer performs the following computation:
 * \f[
 * y = x W + b
 * \f]
 * where:
 * - \f$x\f$ is the input tensor of shape `(batch_size, in_features)`.
 * - \f$W\f$ is the weight matrix of shape `(in_features, out_features)`.
 * - \f$b\f$ is the bias vector of shape `(out_features)`.
 *
 * The weight is initialized using **He initialization**, and the bias follows PyTorch's
 * default uniform initialization.
 */
class Linear {
public:
    /**
     * @brief Weight tensor of the layer.
     *
     * Initialized using He initialization for better training stability.
     */
    Tensor weight;

    /**
     * @brief Bias tensor of the layer.
     *
     * Initialized uniformly based on the number of output features.
     */
    Tensor bias;

    /**
     * @brief Constructs a linear layer with the given input and output dimensions.
     *
     * The weight matrix is initialized using He initialization, and the bias vector
     * is initialized uniformly following PyTorch's convention.
     *
     * @param in_features Number of input features (fan-in).
     * @param out_features Number of output features (fan-out).
     *
     * @example
     * @code
     * Linear layer(128, 64); // Fully connected layer with 128 input and 64 output features
     * @endcode
     */
    Linear(size_t in_features, size_t out_features)
        : weight(Tensor::randn_he(in_features, out_features, true)),
          bias(Tensor::bias_uniform(out_features, true)) {}

    /**
     * @brief Performs a forward pass through the linear layer.
     *
     * Computes the output of the linear transformation:
     * \f[
     * y = x W + b
     * \f]
     * where `x` is the input tensor, `W` is the weight matrix, and `b` is the bias.
     *
     * @param x The input tensor of shape `(batch_size, in_features)`.
     * @return Tensor The output tensor of shape `(batch_size, out_features)`.
     *
     * @example
     * @code
     * Tensor input = Tensor::randn({32, 128}); // Batch of 32 samples, 128 features each
     * Linear layer(128, 64);
     * Tensor output = layer.forward(input); // Output has shape (32, 64)
     * @endcode
     */
    Tensor forward(Tensor x) {
        return x.matmul(weight) + bias;
    }

    /**
     * @brief Sets the layer to evaluation mode.
     *
     * Disables gradient tracking for the weight and bias, typically used during inference.
     *
     * @example
     * @code
     * layer.eval();
     * @endcode
     */
    void eval() {
        weight.eval();
        bias.eval();
    }

    /**
     * @brief Sets the layer to training mode.
     *
     * Re-enables gradient tracking for the weight and bias, typically used during training.
     *
     * @example
     * @code
     * layer.train();
     * @endcode
     */
    void train() {
        weight.train();
        bias.train();
    }

private:
};

#endif // MODELS_H