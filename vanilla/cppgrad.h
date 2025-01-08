#ifndef CPPGRAD_H
#define CPPGRAD_H

#include <vector>
#include <functional>
#include <memory>
#include <stdexcept>

class Tensor {
public:
    std::vector<float> data;
    std::vector<size_t> shape;
    bool requires_grad;
    std::shared_ptr<Tensor> grad;
    std::function<void()> backward_fn;

    // Constructor that infers shape as 1D
    Tensor(const std::vector<float>& data, bool requires_grad = false)
        : data(data), requires_grad(requires_grad), grad(nullptr) {
        // Default shape is 1D: [data.size()]
        shape = { data.size() };
        if (requires_grad) {
            grad = std::make_shared<Tensor>(std::vector<float>(data.size(), 0.0f), false);
            grad->shape = shape;
        }
    }

    // Constructor with explicit shape
    Tensor(const std::vector<float>& data, const std::vector<size_t>& shape, bool requires_grad = false)
        : data(data), shape(shape), requires_grad(requires_grad), grad(nullptr) {
        if (numel(shape) != data.size()) {
            throw std::invalid_argument("Data size does not match shape.");
        }
        if (requires_grad) {
            grad = std::make_shared<Tensor>(std::vector<float>(data.size(), 0.0f), shape, false);
        }
    }

    // Constructor with explicit shape and gradient
    Tensor(const std::vector<float>& data, const std::vector<size_t>& shape, bool requires_grad, std::shared_ptr<Tensor> grad)
        : data(data), shape(shape), requires_grad(requires_grad), grad(grad) {
        if (requires_grad && grad == nullptr) {
            this->grad = std::make_shared<Tensor>(std::vector<float>(data.size(), 0.0f), false);
            this->grad->shape = shape;
        }
    }


    void backward();

    /* binary addition operator */
    Tensor operator+(const Tensor& other) const;
    /* binary minus operator */
    Tensor operator-(const Tensor& other) const;
    /* unary minus operator */
    Tensor operator-() const;
    /* elementwise multiplication operator */
    Tensor operator*(const Tensor& other) const;
    /* matrix multiplication */
    Tensor matmul(const Tensor &other) const;

private:
    static size_t numel(const std::vector<size_t>& shp) {
        size_t product = 1;
        for (auto s : shp) {
            product *= s;
        }
        return product;
    }

    static bool shapes_equal(const std::vector<size_t>& a, const std::vector<size_t>& b) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); i++) {
            if (a[i] != b[i]) return false;
        }
        return true;
    }
};

#endif // CPPGRAD_H
