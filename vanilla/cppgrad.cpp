#include "cppgrad.h"
#include <iostream>

void Tensor::backward() {

    if (this->numel(shape) != 1) {
        throw std::runtime_error("Backward only supported for scalar outputs");
    }

    if (!requires_grad) {
        throw std::runtime_error("This tensor does not require gradient");
    }
    std::cout << "Backward function called" << std::endl;
    // Initialize gradient if necessary
    if (!grad) {
        grad = std::make_shared<Tensor>(std::vector<float>(data.size(), 1.0f), shape, false);
    } else {
        // Ensure the gradient is explicitly set to 1.0 for the target
        std::fill(grad->data.begin(), grad->data.end(), 1.0f);
    }

    if (backward_fn) {
        backward_fn();
    }
}

/* Overloaded + operator */
Tensor Tensor::operator+(const Tensor& other) const{
    std::cout << "Addition operator called" << std::endl;
    if (!shapes_equal(this->shape, other.shape)) {
        throw std::invalid_argument("Tensor shapes must match for addition.");
    }

    // Perform addition
    std::vector<float> result_data(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        result_data[i] = data[i] + other.data[i];
    }

    Tensor result(result_data, shape, requires_grad || other.requires_grad);

    if (result.requires_grad) {
        auto sz = data.size();

        auto this_requires_grad = this->requires_grad;
        auto other_requires_grad = other.requires_grad;

        auto this_grad = this->grad;
        auto other_grad = other.grad;
        auto this_backward_fn = this->backward_fn;
        auto other_backward_fn = other.backward_fn;

        // Capture the result gradients
        auto result_grad = result.grad;

        result.backward_fn = [sz, 
                              this_requires_grad, other_requires_grad,
                              this_grad, other_grad,
                              this_backward_fn, other_backward_fn,
                              result_grad]() {
            // Update this grad
            if (this_requires_grad && this_grad) {
                for (size_t i = 0; i < sz; i++) {
                    this_grad->data[i] += result_grad->data[i];
                }
                if (this_backward_fn) this_backward_fn();
            }

            // Update other grad
            if (other_requires_grad && other_grad) {
                for (size_t i = 0; i < sz; i++) {
                    other_grad->data[i] += result_grad->data[i];
                }
                if (other_backward_fn) other_backward_fn();
            }
        };
    }

    return result;
}

// Unary minus operator
Tensor Tensor::operator-() const {
    /* Negate data */
    std::vector<float> negated_data(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        negated_data[i] = -data[i];
    }
    std::shared_ptr<Tensor> negated_grad = nullptr;

    /* Negated gradient if necessary */
    if (requires_grad) {
        std::vector<float> negated_grad_data(grad->data.size());
        for (size_t i = 0; i < grad->data.size(); i++) {
            negated_grad_data[i] = -grad->data[i];
        }
        negated_grad = std::make_shared<Tensor>(negated_grad_data, grad->requires_grad);
        negated_grad->shape = grad->shape;
    }

    // Create the negated Tensor
    Tensor result(negated_data, shape, requires_grad, negated_grad);

    // Set up the backward function for the unary minus operator
    if (result.requires_grad) {
        auto sz = data.size();            // Size of the tensor
        auto this_requires_grad = this->requires_grad;
        auto this_grad = this->grad;
        auto this_backward_fn = this->backward_fn;
        auto result_grad = result.grad;  // Result's gradient

        result.backward_fn = [sz, this_requires_grad, this_grad, this_backward_fn, result_grad]() {
            if (this_requires_grad && this_grad) {
                // Propagate the negated gradient
                for (size_t i = 0; i < sz; i++) {
                    this_grad->data[i] -= result_grad->data[i];
                }
                // Call the previous backward function if it exists
                if (this_backward_fn) this_backward_fn();
            }
        };
    }

    return result;
}

/* binary minus operator */
Tensor Tensor::operator-(const Tensor& other) const{
    /* check if shapes match */
    if (!shapes_equal(this->shape, other.shape)) {
        throw std::invalid_argument("Tensor shapes must match for subtraction.");
    }

    return *this + (-other);
}

/* Overloaded * operator */
Tensor Tensor::operator*(const Tensor& other) const{
    std::cout << "Multiplication operator called" << std::endl;
    if (!shapes_equal(this->shape, other.shape)) {
        throw std::invalid_argument("Tensor shapes must match for multiplication.");
    }

    // Perform multiplication
    std::vector<float> result_data(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        result_data[i] = data[i] * other.data[i];
    }

    Tensor result(result_data, shape, requires_grad || other.requires_grad);

    if (result.requires_grad) {
        auto sz = data.size();

        auto this_requires_grad = this->requires_grad;
        auto other_requires_grad = other.requires_grad;

        auto this_grad = this->grad;
        auto other_grad = other.grad;
        auto this_backward_fn = this->backward_fn;
        auto other_backward_fn = other.backward_fn;

        auto this_data = this->data;
        auto other_data = other.data;
        auto result_grad = result.grad;

        result.backward_fn = [sz, 
                              this_requires_grad, other_requires_grad,
                              this_grad, other_grad,
                              this_backward_fn, other_backward_fn,
                              this_data, other_data, result_grad]() {
            if (this_requires_grad && this_grad) {
                for (size_t i = 0; i < sz; i++) {
                    this_grad->data[i] += other_data[i] * result_grad->data[i];
                }
                if (this_backward_fn) this_backward_fn();
            }

            if (other_requires_grad && other_grad) {
                for (size_t i = 0; i < sz; i++) {
                    other_grad->data[i] += this_data[i] * result_grad->data[i];
                }
                if (other_backward_fn) other_backward_fn();
            }
        };
    }

    return result;
}

/*
    Matrix multiplication of two tensors, without broadcasting.
    We assume 
    A has shape [b1, b2, ..., bN, m, n]
    B has shape [b1, b2, ..., bN, x, p]
    such that n == x
    thus:
    C = A.matul(B) has shape [b1, b2, ..., bN, m, p]
*/
Tensor Tensor::matmul(const Tensor &other) const{

    /* check matrix dimensions */

    /* num_dim >= 2 for both matrices*/
    if (shape.size() < 2 || other.shape.size() < 2)
        throw std::invalid_argument("Both matrices must have at least 2 columns");

    size_t m = shape[shape.size() - 2];
    size_t n = shape[shape.size() - 1];

    size_t x = other.shape[other.shape.size() - 2];
    size_t p = other.shape[other.shape.size() - 1];

    if (n != x)
        throw std::invalid_argument("Matricies must have matching inner dimensions");
    
    /* check if matrices have the same number of dimensions */
    if (shape.size() != other.shape.size())
        throw std::invalid_argument("Matrices must have the same number of columns, no broadcasting");

    /* check if leading dimensions match */
    std::vector<size_t> batch_shape;
    for (size_t i = 0; i < shape.size() - 2; i++) {
        if (shape[i] != other.shape[i]) {
            throw std::invalid_argument("Batch dimensions must match for bmm (no broadcasting).");
        }
        batch_shape.push_back(shape[i]);
    }

    std::vector<size_t> result_shape = batch_shape;
    result_shape.push_back(m);
    result_shape.push_back(p);

    /* comupte the number of elements */
    size_t total_elems = this->numel(result_shape);
    
    std::vector<float> result_data(total_elems, 0.0f);

    /* forward pass */
    /* triple-nested loop: over each batch, over each i, j, k */

    for (size_t b = 0; b < total_elems/(m*p); b++) {
        // offset in A = b * (m*n)
        // offset in B = b * (n*p)
        // offset in C = b * (m*p)
        size_t A_offset = b * (m * n);
        size_t B_offset = b * (n * p);
        size_t C_offset = b * (m * p);

        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < p; j++) {
                float sum = 0.0f;
                for (size_t k = 0; k < n; k++) {
                    sum += data[A_offset + i*n + k] *
                           other.data[B_offset + k*p + j];
                }
                result_data[C_offset + i*p + j] = sum;
            }
        }
    }

    /* Construct the output tensor */
    bool out_requires_grad = (requires_grad || other.requires_grad);
    Tensor result(result_data, result_shape, out_requires_grad);

    /* Setup the backward_fn if needed */
    if (out_requires_grad) {
        auto this_requires_grad = this->requires_grad;
        auto other_requires_grad = other.requires_grad;

        auto this_grad = this->grad;
        auto other_grad = other.grad;
        auto result_grad = result.grad;

        auto A_data = data;
        auto B_data = other.data;

        auto this_shape = shape;
        auto other_shape = other.shape;
        auto res_shape = result_shape;

        auto this_backward_fn = this->backward_fn;
        auto other_backward_fn = other.backward_fn;

        result.backward_fn = [=]() mutable { // mutable ensures we can change the read only copies in the capture within the lambda            /* Grad wrt A:  dA = dC * B^T  (for each batch) *
            /* Grad wrt B:  dB = A^T * dC */

            if (this_requires_grad && this_grad) {
                for (size_t b = 0; b < total_elems/(m*p); b++) {
                    size_t A_offset = b * (m * n);
                    size_t B_offset = b * (n * p);
                    size_t C_offset = b * (m * p);

                    /* dA shape = [m, n], dC shape = [m, p], B shape = [n, p] */
                    /* dA[i,k] += sum(dC[i,j] * B[k,j]) over j */
                    for (size_t i = 0; i < m; i++) {
                        for (size_t k = 0; k < n; k++) {
                            float grad_value = 0.0f;
                            for (size_t j = 0; j < p; j++) {
                                grad_value += result_grad->data[C_offset + i*p + j]
                                              * B_data[B_offset + k*p + j];
                            }
                            this_grad->data[A_offset + i*n + k] += grad_value;
                        }
                    }
                }
                /* call the old backward_fn if needed */
                if (this_backward_fn) {
                    this_backward_fn();
                }
            }

            if (other_requires_grad && other_grad) {
                for (size_t b = 0; b < total_elems/(m*p); b++) {
                    size_t A_offset = b * (m * n);
                    size_t B_offset = b * (n * p);
                    size_t C_offset = b * (m * p);

                    /* dB shape = [n, p], A shape = [m, n], dC shape = [m, p] */
                    /* dB[k,j] += sum(A[i,k] * dC[i,j]) over i */
                    for (size_t k = 0; k < n; k++) {
                        for (size_t j = 0; j < p; j++) {
                            float grad_value = 0.0f;
                            for (size_t i = 0; i < m; i++) {
                                grad_value += A_data[A_offset + i*n + k]
                                              * result_grad->data[C_offset + i*p + j];
                            }
                            other_grad->data[B_offset + k*p + j] += grad_value;
                        }
                    }
                }
                /* call the old backward_fn if needed */
                if (other_backward_fn) {
                    other_backward_fn();
                }
            }
        };

    }
    return result;
}