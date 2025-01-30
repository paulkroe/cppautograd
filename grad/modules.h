#ifndef MODELS_H
#define MODELS_H

#include "./cppgrad.h"

class Linear {
    public:
        Tensor weight;
        Tensor bias;

        Linear(size_t in_features, size_t out_features)
            : weight(Tensor::randn_he(in_features, out_features, true)),
            bias(Tensor::bias_uniform(out_features, true)) {}


        Tensor forward(Tensor x, const size_t num_threads = 1) {
            return x.matmul(weight, num_threads) + bias;
        }

        void eval() {
            weight.eval();
            bias.eval();
        }

        void train() {
            weight.train();
            bias.train();
        }

    private:
};

#endif // MODELS_H