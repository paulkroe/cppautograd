#ifndef MODELS_H
#define MODELS_H

#include "./cppgrad.h"

class Linear {
    public:
        Tensor weight;
        Tensor bias;

        Linear(size_t in_features, size_t out_features)
            : weight(Tensor::randn({in_features, out_features}, true)),
              bias(Tensor::randn({out_features}, true)) {}

        Tensor forward(Tensor x) {
            return x.matmul(weight) + bias;
        }
    private:
};

#endif // MODELS_H