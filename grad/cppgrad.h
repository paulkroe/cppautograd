#ifndef CPPGRAD_H
#define CPPGRAD_H

#include <atomic>
#include <thread>
#include <mutex>
#include <functional>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <random>
#include <cmath>
#include <vector>
#include <stack>
#include <unordered_set>

/* global tensor id */
extern std::atomic<std::uint64_t> id_counter;
/* helper function to get tensor id*/
size_t get_id();

/* global random number generator */
extern std::mt19937 global_generator;
/* helper function to set random seed */
void set_seed(int seed);

/* Utility functions: */

/* 
 * given two shapes, returns the shape that results 
 * from broadcasting the two shapes
 * returns an empty vector if the shapes are not broadcastable
 * when iterating over the dimension sizes, starting at the trailing dimension,
 * the dimension sizes must either be equal, one of them is 1, or one of them does not exist
 */
std::vector<size_t> broadcast(const std::vector<size_t>& shape1, const std::vector<size_t>& shape2);

/*
 * given a multi-dimensional coordinate and a shape, flatten 
 * back to a single offset in row-major order
 */
size_t ravel_index(const std::vector<size_t>& coords, const std::vector<size_t>& shape);

/*
 * given a linear index `idx` and a shape vector, produce
 * the multi-dimensional coordinate
 * e.g. shape=[2,3,4], idx in [0..23]
 * => coord=[b,c,d]
 */
std::vector<size_t> unravel_index(size_t idx, const std::vector<size_t>& shape);

/*
 * reduce the gradient by summing over broadcasted dimension
 */
std::vector<float> reduce_grad(const std::vector<float>& grad, 
        const std::vector<size_t>& grad_shape, 
        const std::vector<size_t>& original_shape);

/* helper function to count the number of elements */
size_t numel(const std::vector<size_t>& shp);

/* helper function to print tensor shape */
void printShape(const std::vector<size_t>& shape);

/* hel[er function to print two tensor shapes */
void printShapes(const std::vector<size_t>& shape1, const std::vector<size_t>& shape2);

class TensorData {
public:
    /* Global mutex for thread_gradients */
    static std::mutex GLOBAL_GRAD_MUTEX;      
    /* Global mutex for parents */
    static std::mutex GLOBAL_PARENTS_MUTEX;

    size_t id;
    std::vector<float> data;
    std::vector<size_t> shape;
    bool requires_grad;
    mutable std::unordered_map<std::thread::id, std::shared_ptr<TensorData>> thread_gradients;
    std::function<void()> backward_fn;
    mutable std::unordered_map<std::thread::id, std::unordered_set<std::shared_ptr<TensorData>>> parents;

    /* constructor inferring tensor shape to be 1D */
    TensorData(const std::vector<float>& data, bool requires_grad = false)
        : data(data), requires_grad(requires_grad) { 
            shape = { data.size() };
            id = get_id();

            if (requires_grad) {
                std::thread::id tid = std::this_thread::get_id();
                std::lock_guard<std::mutex> lock(GLOBAL_GRAD_MUTEX);
                thread_gradients[tid] = std::make_shared<TensorData>(std::vector<float>(data.size(), 0.0f), shape, false);
            }
    }

    /* constructor creating a tensor with explicit shape */
    TensorData(const std::vector<float>& data, const std::vector<size_t>& shape, bool requires_grad = false)
        : data(data), shape(shape), requires_grad(requires_grad) {
            /* check if shape matches */
            if (numel(shape) != data.size()) {
                throw std::invalid_argument("Data size does not match shape.");
            }
            id = get_id();

            if (requires_grad) {
                std::thread::id tid = std::this_thread::get_id();
                std::lock_guard<std::mutex> lock(GLOBAL_GRAD_MUTEX);
                thread_gradients[tid] = std::make_shared<TensorData>(std::vector<float>(data.size(), 0.0f), shape, false);
            }
    }

    /* constructor creating a tensor with an explicit shape and gradient */
    TensorData(const std::vector<float>& data, const std::vector<size_t>& shape, bool requires_grad, std::shared_ptr<TensorData> grad)
        : data(data), shape(shape), requires_grad(requires_grad) {
            /* check if shape matches */
            if (numel(shape) != data.size()) {
                throw std::invalid_argument("Data size does not match shape.");
            }
            id = get_id();

            if (requires_grad) {
                std::thread::id tid = std::this_thread::get_id();
                std::lock_guard<std::mutex> lock(GLOBAL_GRAD_MUTEX);
                if (!grad) {
                    grad = std::make_shared<TensorData>(std::vector<float>(data.size(), 0.0f), shape, false);
                }
                thread_gradients[tid] = grad;
            }
    }

   private:
};

class Tensor {
    public:

    std::shared_ptr<TensorData> ptr;

    Tensor(const std::vector<float>& data, bool requires_grad = false) {
        this->ptr = std::make_shared<TensorData>(data, requires_grad);
    }

    Tensor(const std::vector<float>& data, const std::vector<size_t>& shape, bool requires_grad = false) {
        this->ptr = std::make_shared<TensorData>(data, shape, requires_grad);
    }

    Tensor(const std::vector<float>& data, const std::vector<size_t>& shape, bool requires_grad, std::shared_ptr<TensorData> grad) {
        this->ptr = std::make_shared<TensorData>(data, shape, requires_grad, grad);
    }

    Tensor(std::shared_ptr<TensorData> ptr) : ptr(ptr) {}

    Tensor() : ptr(NULL) {}

    /* backward function */
    void backward();

    /* binary addition operator */
    Tensor operator+(const Tensor& other) const;
    Tensor operator+(const float other) const;
    /* binary minus operator */
    Tensor operator-(const Tensor& other) const;
    Tensor operator-(const float other) const;
    /* unary minus operator */
    Tensor operator-() const;
    /* elementwise multiplication operator */
    Tensor operator*(const Tensor& other) const;
    Tensor operator*(const float other) const;
    // /* elementwise division operator */
    Tensor operator/(const Tensor& other) const;
    Tensor operator/(const float other) const;
    /* overload the << operator */
    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
    /* matrix multiplication */
    Tensor matmul(const Tensor &other) const;
    /* sum over given dimension */
    Tensor sum(const size_t dim) const;
    /* sum over trailing dimension */
    Tensor sum() const;
    /* mean over given dimension */
    Tensor mean(const size_t dim) const;
    /* mean over trailing dimension */
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

    /* initialization functions: */

    /* 
     * initialize a random tensor
     * each element is sampled from U[0, 1]
     */
    static Tensor randn(const std::vector<size_t>& shape, bool requires_grad); 

    /*
     * initialize a random tensor using He initialization
     * sampled from a uniform distribution scaled by stddev.
     */
    static Tensor randn_he(size_t in_features, size_t out_features, bool requires_grad);
        
    /*
     * initialize a bias tensor with values sampled uniformly from [-bound, bound],
     * where bound = 1 / sqrt(in_features), following PyTorch's bias initialization.
     */
    static Tensor bias_uniform(size_t in_features, bool requires_grad);

    // /* helper functions: */

    /* helper function zeroing out the gradient */
    void zero_grad();
    /* helper function to print the shape of a tensor */
    void print_shape() const; 
    /* helper function to print a tensor */
    void print_recursive(std::ostream& os, size_t dim, size_t offset, size_t stride) const; 
    /* helper function returning the shape of a tensor */
    std::vector<size_t> shape() const;
    /* helper function returning the data of a tensor */
    std::vector<float> data() const;
    /* helper function returning the gradient tensor */
    Tensor grad() const;
    /* disable gradient computation */
    void eval();
    /* enable gradient computation */
    void train();

    private:
};

/* Loss functions: */

/* 
 * Cross Entropy Loss
 * y_pred: Tensor of shape (batch_size, num_classes)
 * y_true: Tensor of shape (batch_size)
 */
Tensor CrossEntropyLoss(const Tensor& y_pred, const Tensor& y_true);

#endif // CPPGRAD_H