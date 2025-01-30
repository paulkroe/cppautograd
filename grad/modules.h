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


        Tensor forward(Tensor x) {
            return x.matmul(weight) + bias;
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