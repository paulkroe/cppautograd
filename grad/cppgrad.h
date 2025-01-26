#ifndef CPPGRAD_H
#define CPPGRAD_H

#include <functional>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <random>
#include <cmath>
#include <vector>
#include <stack>
#include <unordered_set>

int get_id();
/* Global random number generator */
extern std::mt19937 global_generator;
void set_seed(int seed);

// class Tensor {
class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    int id = -1;
    std::vector<float> data;
    std::vector<size_t> shape;
    bool requires_grad;
    std::shared_ptr<Tensor> grad;
    std::function<void()> backward_fn;
    std::vector<std::shared_ptr<Tensor>> parents;

    // Constructor that infers shape as 1D
    Tensor(const std::vector<float>& data, bool requires_grad = false)
        : data(data), requires_grad(requires_grad), grad(nullptr) {
        // Default shape is 1D: [data.size()]
        shape = { data.size() };
        if (requires_grad) {
            grad = std::make_shared<Tensor>(std::vector<float>(data.size(), 0.0f), false);
            grad->shape = shape;
        }
        id = get_id();
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
        id = get_id();
    }

    // Constructor with explicit shape and gradient
    Tensor(const std::vector<float>& data, const std::vector<size_t>& shape, bool requires_grad, std::shared_ptr<Tensor> grad)
        : data(data), shape(shape), requires_grad(requires_grad), grad(grad) {
        if (requires_grad && grad == nullptr) {
            this->grad = std::make_shared<Tensor>(std::vector<float>(data.size(), 0.0f), false);
            this->grad->shape = shape;
        }
        id = get_id();
    }
    
    void print_recursive(std::ostream& os, size_t dim, size_t offset, size_t stride) const;
    // Overload the << operator
    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

    /* print shape */
    void print_shape() const{
        for (size_t s : shape) {
            std::cout << s << " ";
        }
        std::cout << std::endl;
    }

    /* zero out the gradient */
    void zero_grad() {
        if (grad) {
            std::fill(grad->data.begin(), grad->data.end(), 0.0f);
        }
    }

    /* random tensor */
    static Tensor randn(const std::vector<size_t>& shape, bool requires_grad = false) {
        size_t total_elems = numel(shape);
        std::vector<float> data(total_elems);

        // Use the global random generator
        std::uniform_real_distribution<float> distribution(0.0, 1.0);
        for (size_t i = 0; i < total_elems; i++) {
            data[i] = distribution(global_generator);
        }

        return Tensor(data, shape, requires_grad);
    }

    static Tensor randn_he(size_t in_features, size_t out_features, bool requires_grad = false) {
        float stddev = std::sqrt(2.0f / in_features);
        std::vector<float> data(in_features * out_features);
        for (size_t i = 0; i < data.size(); i++) {
            data[i] = stddev * ((float)rand() / RAND_MAX * 2 - 1); // Uniform [-stddev, stddev]
        }
        return Tensor(data, {in_features, out_features}, requires_grad);
    }

    static Tensor bias_uniform(size_t in_features, bool requires_grad = false) {
        float bound = 1.0f / std::sqrt(in_features);  // PyTorch-style bias init
        std::vector<float> data(in_features);
        for (size_t i = 0; i < data.size(); i++) {
            data[i] = ((float)rand() / RAND_MAX * 2 - 1) * bound;  // Uniform[-bound, bound]
        }
        return Tensor(data, {in_features}, requires_grad);
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
    /* elementwise division operator */
    Tensor operator/(const Tensor& other) const;
    /* matrix multiplication */
    Tensor matmul(const Tensor &other) const;
    /* sum over dimension */
    Tensor sum(size_t dim) const;
    Tensor sum() const;
    /* mean over dimension */
    Tensor mean(size_t dim) const;
    Tensor mean() const;
    /* exp tensor */
    Tensor exp() const;
    /* log tensor */
    Tensor log() const;
    /* softmax over dimension */
    Tensor softmax(size_t dim) const;
    /* softmax */
    Tensor softmax() const;
    /* one hot-encode */
    Tensor onehot_encode(size_t num_classes) const;
    /* activation function */
    Tensor relu() const;

    static size_t numel(const std::vector<size_t>& shp) {
        size_t product = 1;
        for (auto s : shp) {
            product *= s;
        }
        return product;
    }
 
    static std::vector<float> reduce_grad(const std::vector<float>& grad, 
        const std::vector<size_t>& grad_shape, 
        const std::vector<size_t>& original_shape);

private:
    

    // ravel_index: given a multi-dimensional coordinate and a shape, flatten 
    // back to a single offset in row-major order.
    static size_t ravel_index(const std::vector<size_t>& coords, const std::vector<size_t>& shape) {
        size_t idx = 0;
        for (size_t i = 0; i < shape.size(); i++) {
            idx = idx * shape[i] + coords[i];
        }
        return idx;
    } 
    
    static bool shapes_equal(const std::vector<size_t>& a, const std::vector<size_t>& b) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); i++) {
            if (a[i] != b[i]) return false;
        }
        return true;
    }


};

Tensor CrossEntropyLoss(const Tensor& y_pred, const Tensor& y_true);


/* Utility functions */

std::vector<size_t> broadcast(const std::vector<size_t>& shape1, const std::vector<size_t>& shape2);

void printShape(const std::vector<size_t>& shape);
void printShapes(const std::vector<size_t>& shape1, const std::vector<size_t>& shape2);

size_t map_index(const std::vector<size_t>& multi_index,
                        const std::vector<size_t>& shape); 

// unravel_index: given a linear index `idx` and a shape vector, produce
// the multi-dimensional coordinate, e.g. shape=[2,3,4], idx in [0..23] 
// => coord=[b,c,d].
std::vector<size_t> unravel_index(size_t idx, const std::vector<size_t>& shape);
#endif // CPPGRAD_H