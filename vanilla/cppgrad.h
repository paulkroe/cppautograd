#ifndef CPPGRAD_H
#define CPPGRAD_H

#include <vector>
#include <functional>
#include <memory>
#include <stdexcept>
#include <iostream>

int get_id();

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

    // Overload the << operator
    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
        os << "Data: ";
        for (float d : tensor.data) {
            os << d << " ";
        }
        os << "\n";

        os << "Shape: ";
        for (size_t s : tensor.shape) {
            os << s << " ";
        }
        os << "\n";

        if (tensor.requires_grad && tensor.grad) {
            os << "Gradient: ";
            for (float g : tensor.grad->data) {
                os << g << " ";
            }
            os << "\n";
        }

        return os;
    }

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
        for (size_t i = 0; i < total_elems; i++) {
            data[i] = (float)rand() / RAND_MAX;
        }
        return Tensor(data, shape, requires_grad);
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

private:
    // unravel_index: given a linear index `idx` and a shape vector, produce
    // the multi-dimensional coordinate, e.g. shape=[2,3,4], idx in [0..23] 
    // => coord=[b,c,d].
    static std::vector<size_t> unravel_index(size_t idx, const std::vector<size_t>& shape) {
        std::vector<size_t> coords(shape.size());
        for (int i = (int)shape.size() - 1; i >= 0; --i) {
            coords[i] = idx % shape[i];
            idx /= shape[i];
        }
        return coords;
    }

    // ravel_index: given a multi-dimensional coordinate and a shape, flatten 
    // back to a single offset in row-major order.
    static size_t ravel_index(const std::vector<size_t>& coords, const std::vector<size_t>& shape) {
        size_t idx = 0;
        for (size_t i = 0; i < shape.size(); i++) {
            idx = idx * shape[i] + coords[i];
        }
        return idx;
    }

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

    static size_t map_index(const std::vector<size_t>& multi_index,
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


static std::vector<size_t> unflatten_index(size_t flat_index, const std::vector<size_t>& shape);

static std::vector<float> reduce_grad(const std::vector<float>& grad, 
    const std::vector<size_t>& grad_shape, 
    const std::vector<size_t>& original_shape);



};

Tensor CrossEntropyLoss(const Tensor& y_pred, const Tensor& y_true);
#endif // CPPGRAD_H