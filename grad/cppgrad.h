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

/**
 * @brief Global atomic counter for assigning unique tensor IDs.
 *
 * Each new tensor gets a unique ID to track dependencies in the computation graph.
 */
extern std::atomic<std::uint64_t> id_counter;

/**
 * @brief Generates a unique ID for a new tensor.
 *
 * @return size_t A unique tensor ID.
 * @throws std::runtime_error If the ID counter overflows.
 */
size_t get_id();

/**
 * @brief Global random number generator used for weight initialization.
 *
 * The generator is initialized with a fixed seed (`42`) for reproducibility.
 */
extern std::mt19937 global_generator;

/**
 * @brief Sets the random seed for the global random number generator.
 *
 * @param seed The seed value for deterministic random number generation.
 */
void set_seed(int seed);

/* Utility functions: */

/**
 * @brief Computes the broadcasted shape for two tensors.
 *
 * This function follows **NumPy broadcasting rules**:
 * - Two dimensions are compatible if they are equal or one of them is `1`.
 * - If incompatible, an empty vector is returned.
 *
 * @param shape1 The shape of the first tensor.
 * @param shape2 The shape of the second tensor.
 * @return std::vector<size_t> The broadcasted shape, or an empty vector if broadcasting fails.
 */
std::vector<size_t> broadcast(const std::vector<size_t>& shape1, const std::vector<size_t>& shape2);

/**
 * @brief Converts a multi-dimensional index into a single linear index.
 *
 * This function assumes **row-major ordering**.
 *
 * @param multi_index A vector representing a multi-dimensional index.
 * @param shape The shape of the tensor.
 * @return size_t The corresponding linear index.
 */
size_t ravel_index(const std::vector<size_t>& coords, const std::vector<size_t>& shape);

/**
 * @brief Converts a linear index into a multi-dimensional index.
 *
 * @param idx The linear index.
 * @param shape The shape of the tensor.
 * @return std::vector<size_t> The corresponding multi-dimensional index.
 */
std::vector<size_t> unravel_index(size_t idx, const std::vector<size_t>& shape);

/**
 * @brief Reduces a gradient tensor to match the original shape after broadcasting.
 *
 * This function accumulates gradients along broadcasted dimensions to maintain
 * correct gradient propagation.
 *
 * @param grad The gradient tensor after the operation.
 * @param grad_shape The shape of the gradient tensor.
 * @param original_shape The original shape of the tensor before broadcasting.
 * @return std::vector<float> The reduced gradient tensor.
 */
std::vector<float> reduce_grad(const std::vector<float>& grad, 
        const std::vector<size_t>& grad_shape, 
        const std::vector<size_t>& original_shape);

/**
 * @brief Computes the total number of elements in a given shape.
 *
 * @param shp The shape vector representing tensor dimensions.
 * @return size_t The total number of elements.
 */
size_t numel(const std::vector<size_t>& shp);

/**
 * @brief Prints the shape of a tensor to standard output.
 */
void printShape(const std::vector<size_t>& shape);

/**
 * @brief Prints two tensor shapes side by side for debugging.
 *
 * @param shape1 The shape of the first tensor.
 * @param shape2 The shape of the second tensor.
 */
void printShapes(const std::vector<size_t>& shape1, const std::vector<size_t>& shape2);

/**
 * @brief Internal representation of tensor data, supporting autograd and multi-threaded execution.
 *
 * This class encapsulates tensor data, its shape, gradient tracking, and the computation graph
 * necessary for automatic differentiation. Each tensor maintains:
 * - A unique `id` for tracking dependencies.
 * - A `data` vector storing its values.
 * - A `shape` vector defining its dimensions.
 * - A `requires_grad` flag to indicate if gradients should be computed.
 * - A `backward_fn` function for propagating gradients during backpropagation.
 * - Thread-local storage (`thread_gradients`) for per-thread gradient tracking.
 * - A `parents` map storing parent tensors in the computation graph for autograd.
 *
 * The class ensures thread safety using global mutexes (`GLOBAL_GRAD_MUTEX` and `GLOBAL_PARENTS_MUTEX`).
 */
class TensorData {
public:
    /**
     * @brief Global mutex for synchronizing access to `thread_gradients`.
     */
    static std::mutex GLOBAL_GRAD_MUTEX;      

    /**
     * @brief Global mutex for synchronizing access to `parents` in the computation graph.
     */
    static std::mutex GLOBAL_PARENTS_MUTEX;

    size_t id;  ///< Unique identifier for this tensor instance.
    std::vector<float> data;  ///< Tensor values stored in a contiguous vector.
    std::vector<size_t> shape;  ///< Shape of the tensor.
    bool requires_grad;  ///< Indicates whether the tensor participates in autograd.
    
    /**
     * @brief Thread-local gradient storage for multi-threaded backpropagation.
     *
     * Each thread maintains its own gradient tensor to prevent data races.
     */
    mutable std::unordered_map<std::thread::id, std::shared_ptr<TensorData>> thread_gradients;

    /**
     * @brief Backward function for autograd.
     *
     * If `requires_grad` is `true`, this function is used to propagate gradients
     * during backpropagation.
     */
    std::function<void()> backward_fn;

    /**
     * @brief Parent tensors in the autograd computation graph.
     *
     * Stores the parent tensors for each thread, allowing for proper gradient propagation.
     */
    mutable std::unordered_map<std::thread::id, std::unordered_set<std::shared_ptr<TensorData>>> parents;

    /**
     * @brief Constructs a tensor with inferred shape (1D vector).
     *
     * If `requires_grad` is `true`, the tensor is registered for gradient tracking.
     *
     * @param data A vector containing the tensor elements.
     * @param requires_grad If `true`, the tensor participates in backpropagation.
     */
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

    /**
     * @brief Constructs a tensor with an explicit shape.
     *
     * Ensures that the shape matches the data size and registers the tensor for
     * gradient tracking if `requires_grad` is `true`.
     *
     * @param data A vector containing the tensor elements.
     * @param shape The explicit shape of the tensor.
     * @param requires_grad If `true`, the tensor participates in backpropagation.
     * @throws std::invalid_argument If the number of elements in `data` does not match `shape`.
     */
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

    /**
     * @brief Constructs a tensor with an explicit shape and gradient storage.
     *
     * If `requires_grad` is `true`, this constructor initializes a thread-local
     * gradient tensor and associates it with the current thread.
     *
     * @param data A vector containing the tensor elements.
     * @param shape The explicit shape of the tensor.
     * @param requires_grad If `true`, the tensor participates in backpropagation.
     * @param grad A shared pointer to an existing gradient tensor (optional).
     * @throws std::invalid_argument If the number of elements in `data` does not match `shape`.
     */
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

/**
 * @brief Represents a multi-dimensional tensor with automatic differentiation support.
 *
 * The `Tensor` class acts as a wrapper around `TensorData`, providing a high-level API
 * for tensor operations. It supports gradient tracking and integrates with the computation
 * graph for automatic differentiation.
 *
 */
class Tensor {
public:
    /**
     * @brief Shared pointer to the underlying tensor data.
     *
     * This pointer allows multiple `Tensor` instances to share the same underlying data,
     * enabling efficient memory management and computation graph tracking.
     */
    std::shared_ptr<TensorData> ptr;

    /**
     * @brief Constructs a 1D tensor with inferred shape.
     *
     * This constructor initializes a tensor with a 1D shape inferred from `data.size()`.
     * If `requires_grad` is `true`, the tensor participates in automatic differentiation.
     *
     * @param data A vector containing the tensor elements.
     * @param requires_grad If `true`, enables gradient computation for this tensor.
     *
     * @example
     * @code
     * Tensor t({1.0, 2.0, 3.0}, true); // 1D tensor with 3 elements, requires gradient
     * @endcode
     */
    Tensor(const std::vector<float>& data, bool requires_grad = false) {
        this->ptr = std::make_shared<TensorData>(data, requires_grad);
    }

    /**
     * @brief Constructs a tensor with an explicit shape.
     *
     * Initializes a tensor with the specified shape and values. Ensures that the shape
     * matches the number of elements in `data`.
     *
     * @param data A vector containing the tensor elements.
     * @param shape The shape of the tensor.
     * @param requires_grad If `true`, enables gradient computation for this tensor.
     * @throws std::invalid_argument If the number of elements in `data` does not match `shape`.
     *
     * @example
     * @code
     * Tensor t({1.0, 2.0, 3.0, 4.0}, {2, 2}, true); // 2x2 tensor, requires gradient
     * @endcode
     */
    Tensor(const std::vector<float>& data, const std::vector<size_t>& shape, bool requires_grad = false) {
        this->ptr = std::make_shared<TensorData>(data, shape, requires_grad);
    }

    /**
     * @brief Constructs a tensor with an explicit shape and a custom gradient tensor.
     *
     * This constructor allows setting both shape and an external gradient storage,
     * which is useful for custom gradient management.
     *
     * @param data A vector containing the tensor elements.
     * @param shape The shape of the tensor.
     * @param requires_grad If `true`, enables gradient computation for this tensor.
     * @param grad A shared pointer to an external gradient tensor.
     * @throws std::invalid_argument If the number of elements in `data` does not match `shape`.
     *
     * @example
     * @code
     * auto grad = std::make_shared<TensorData>(std::vector<float>(4, 0.0f), {2, 2}, false);
     * Tensor t({1.0, 2.0, 3.0, 4.0}, {2, 2}, true, grad);
     * @endcode
     */
    Tensor(const std::vector<float>& data, const std::vector<size_t>& shape, bool requires_grad, std::shared_ptr<TensorData> grad) {
        this->ptr = std::make_shared<TensorData>(data, shape, requires_grad, grad);
    }

    /**
     * @brief Constructs a tensor from an existing shared pointer to `TensorData`.
     *
     * This constructor allows creating a `Tensor` that shares ownership of an existing `TensorData` object.
     *
     * @param ptr A shared pointer to an existing `TensorData` instance.
     *
     * @example
     * @code
     * auto data = std::make_shared<TensorData>(std::vector<float>{1.0, 2.0, 3.0}, false);
     * Tensor t(data);
     * @endcode
     */
    Tensor(std::shared_ptr<TensorData> ptr) : ptr(ptr) {}

    /**
     * @brief Default constructor initializing an empty tensor.
     *
     * Creates a `Tensor` instance with a `NULL` data pointer.
     *
     * @example
     * @code
     * Tensor empty_tensor;
     * @endcode
     */
    Tensor() : ptr(NULL) {}

    /* backward function */
    void backward();

    /**
     * @brief Element-wise addition of two tensors with broadcasting support.
     *
     * This operator performs element-wise addition between the current tensor and `other`,
     * supporting **broadcasting** to handle different shapes. The resulting tensor's shape
     * follows standard **NumPy broadcasting rules**.
     *
     * Steps:
     * 1. **Check broadcast compatibility:** Ensures the two tensors can be broadcasted.
     * 2. **Iterate over result data:** Performs element-wise addition with correct indexing.
     * 3. **Construct multi-dimensional indices:** Maps input indices to the broadcasted result.
     * 4. **Compute sum:** Stores the computed sum in the output tensor.
     * 5. **Set up autograd:** If either tensor requires gradients, registers a backward function.
     *
     * @param other The tensor to add to the current tensor.
     * @return Tensor The result of element-wise addition, with shape determined by broadcasting.
     * 
     * @throws std::invalid_argument If the tensors cannot be broadcasted.
     *
     * @note If either operand has `requires_grad = true`, the resulting tensor will also
     *       require gradients and track operations for automatic differentiation.
     *
     * @example
     * @code
     * Tensor a({1, 2, 3}, {3}, true);   // Shape: (3,)
     * Tensor b({5}, {1}, false);        // Shape: (1,)
     * Tensor c = a + b;                 // Shape: (3,), values: {6, 7, 8}
     * @endcode
     */
    Tensor operator+(const Tensor& other) const;

    /**
     * @brief Adds a scalar value to each element of the tensor.
     *
     * This operator performs element-wise addition between the current tensor and a scalar.
     * It internally converts the scalar into a **broadcastable tensor** before performing the addition.
     *
     * @param other The scalar value to add to each element of the tensor.
     * @return Tensor The result of adding `other` to each element of the tensor.
     *
     * @note If the tensor has `requires_grad = true`, the resulting tensor will also
     *       require gradients and track operations for automatic differentiation.
     *
     * @example
     * @code
     * Tensor a({1.0, 2.0, 3.0}, {3}, true); // Shape: (3,)
     * Tensor b = a + 5.0f;                  // Shape: (3,), values: {6.0, 7.0, 8.0}
     * @endcode
     */
    Tensor operator+(const float other) const;
    
    /**
     * @brief Performs element-wise subtraction between two tensors with broadcasting support.
     *
     * This operator computes the difference between the current tensor and `other`, supporting
     * **broadcasting** as per **NumPy broadcasting rules**. Internally, it utilizes the overloaded
     * unary negation and addition operators:
     * \f[
     * \text{result} = \text{this} - \text{other} = \text{this} + (-\text{other})
     * \f]
     *
     * @param other The tensor to subtract from the current tensor.
     * @return Tensor The result of element-wise subtraction, with shape determined by broadcasting.
     *
     * @throws std::invalid_argument If the tensors cannot be broadcasted.
     *
     * @note If either operand has `requires_grad = true`, the resulting tensor will also
     *       require gradients and track operations for automatic differentiation.
     *
     * @example
     * @code
     * Tensor a({1, 2, 3}, {3}, true);   // Shape: (3,)
     * Tensor b({5, 1, 0}, {3}, false);  // Shape: (3,)
     * Tensor c = a - b;                 // Shape: (3,), values: {-4, 1, 3}
     * @endcode
     */
    Tensor operator-(const Tensor& other) const;

    /**
     * @brief Subtracts a scalar value from each element of the tensor.
     *
     * This operator performs element-wise subtraction between the tensor and a scalar.
     * Internally, it converts the scalar into a **broadcastable tensor** before performing the operation.
     *
     * @param other The scalar value to subtract from each element of the tensor.
     * @return Tensor The result of subtracting `other` from each element of the tensor.
     *
     * @note If the tensor has `requires_grad = true`, the resulting tensor will also
     *       require gradients and track operations for automatic differentiation.
     *
     * @example
     * @code
     * Tensor a({1.0, 2.0, 3.0}, {3}, true); // Shape: (3,)
     * Tensor b = a - 2.0f;                  // Shape: (3,), values: {-1.0, 0.0, 1.0}
     * @endcode
     */
    Tensor operator-(const float other) const;
    
    /**
     * @brief Negates all elements in the tensor.
     *
     * This operator applies element-wise negation to the tensor, returning a new tensor
     * where each element is multiplied by `-1`.
     *
     * Steps:
     * 1. **Negate the data:** Computes `-x` for each element.
     * 2. **Negate the gradient (if applicable):** Ensures correct gradient propagation.
     * 3. **Construct the result tensor:** Stores the negated values.
     * 4. **Set up autograd:** If `requires_grad = true`, registers a backward function.
     *
     * @return Tensor A tensor with all elements negated.
     *
     * @note If the tensor has `requires_grad = true`, the resulting tensor will also
     *       require gradients and track operations for automatic differentiation.
     *
     * @example
     * @code
     * Tensor a({1.0, -2.0, 3.0}, {3}, true); // Shape: (3,)
     * Tensor b = -a;                         // Shape: (3,), values: {-1.0, 2.0, -3.0}
     * @endcode
     */
    Tensor operator-() const;

    /**
     * @brief Performs element-wise multiplication between two tensors with broadcasting support.
     *
     * This operator computes the product of the current tensor and `other`, supporting
     * **broadcasting** to handle different shapes as per **NumPy broadcasting rules**.
     *
     * Steps:
     * 1. **Check broadcast compatibility:** Ensures the two tensors can be broadcasted.
     * 2. **Iterate over result data:** Computes the product for each element.
     * 3. **Construct multi-dimensional indices:** Maps input indices to the broadcasted result.
     * 4. **Compute product:** Stores the computed product in the output tensor.
     * 5. **Set up autograd:** If either tensor requires gradients, registers a backward function.
     *
     * @param other The tensor to multiply with the current tensor.
     * @return Tensor The result of element-wise multiplication, with shape determined by broadcasting.
     *
     * @throws std::invalid_argument If the tensors cannot be broadcasted.
     *
     * @note If either operand has `requires_grad = true`, the resulting tensor will also
     *       require gradients and track operations for automatic differentiation.
     *
     * @example
     * @code
     * Tensor a({1, 2, 3}, {3}, true);   // Shape: (3,)
     * Tensor b({5, 1, 0}, {3}, false);  // Shape: (3,)
     * Tensor c = a * b;                 // Shape: (3,), values: {5, 2, 0}
     * @endcode
     */
    Tensor operator*(const Tensor& other) const;

    /**
     * @brief Multiplies each element of the tensor by a scalar value.
     *
     * This operator performs element-wise multiplication between the tensor and a scalar.
     * Internally, it converts the scalar into a **broadcastable tensor** before performing
     * the operation.
     *
     * @param other The scalar value to multiply with each element of the tensor.
     * @return Tensor The result of multiplying `other` with each element of the tensor.
     *
     * @note If the tensor has `requires_grad = true`, the resulting tensor will also
     *       require gradients and track operations for automatic differentiation.
     *
     * @example
     * @code
     * Tensor a({1.0, 2.0, 3.0}, {3}, true); // Shape: (3,)
     * Tensor b = a * 2.0f;                  // Shape: (3,), values: {2.0, 4.0, 6.0}
     * @endcode
     */
    Tensor operator*(const float other) const;

    /**
     * @brief Performs element-wise division between two tensors with broadcasting support.
     *
     * This operator computes the division of the current tensor by `other`, supporting
     * **broadcasting** as per **NumPy broadcasting rules**.
     *
     * Steps:
     * 1. **Check broadcast compatibility:** Ensures the two tensors can be broadcasted.
     * 2. **Iterate over result data:** Computes the quotient for each element.
     * 3. **Construct multi-dimensional indices:** Maps input indices to the broadcasted result.
     * 4. **Compute division:** Stores the computed quotient in the output tensor.
     * 5. **Set up autograd:** If either tensor requires gradients, registers a backward function.
     *
     * @param other The tensor by which to divide the current tensor.
     * @return Tensor The result of element-wise division, with shape determined by broadcasting.
     *
     * @throws std::invalid_argument If the tensors cannot be broadcasted.
     * @throws std::runtime_error If division by zero occurs.
     *
     * @note If either operand has `requires_grad = true`, the resulting tensor will also
     *       require gradients and track operations for automatic differentiation.
     *
     * @example
     * @code
     * Tensor a({10, 20, 30}, {3}, true);  // Shape: (3,)
     * Tensor b({2, 5, 3}, {3}, false);    // Shape: (3,)
     * Tensor c = a / b;                   // Shape: (3,), values: {5.0, 4.0, 10.0}
     * @endcode
     */
    Tensor operator/(const Tensor& other) const;

    /**
     * @brief Divides each element of the tensor by a scalar value.
     *
     * This operator performs element-wise division between the tensor and a scalar.
     * Internally, it converts the scalar into a **broadcastable tensor** before performing
     * the operation.
     *
     * @param other The scalar value by which to divide each element of the tensor.
     * @return Tensor The result of dividing each element of the tensor by `other`.
     *
     * @throws std::runtime_error If division by zero occurs.
     *
     * @note If the tensor has `requires_grad = true`, the resulting tensor will also
     *       require gradients and track operations for automatic differentiation.
     *
     * @example
     * @code
     * Tensor a({10.0, 20.0, 30.0}, {3}, true); // Shape: (3,)
     * Tensor b = a / 2.0f;                     // Shape: (3,), values: {5.0, 10.0, 15.0}
     * @endcode
     */
    Tensor operator/(const float other) const;

    /**
     * @brief Overloaded output stream operator for printing tensors.
     *
     * This operator prints the tensor in a format similar to PyTorch:
     * ```
     * Tensor([[1.0, 2.0], [3.0, 4.0]], shape=[2, 2])
     * ```
     * The tensor values are printed using `print_recursive()` to ensure correct formatting
     * for multi-dimensional tensors.
     *
     * @param os The output stream to write to.
     * @param tensor The tensor to be printed.
     * @return std::ostream& The output stream with the formatted tensor data.
     *
     * @example
     * @code
     * Tensor t({1.0, 2.0, 3.0, 4.0}, {2, 2});
     * std::cout << t; // Prints: Tensor([[1.0, 2.0], [3.0, 4.0]], shape=[2, 2])
     * @endcode
     */
    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
    
    /**
     * @brief Performs batched matrix multiplication with broadcasting support.
     *
     * This function computes the matrix multiplication C = A.matmul(B), supporting batches and broadcasting.
     * It assumes that A has shape [..., m, n] and B has shape [..., x, p] with n == x, resulting in 
     * an output tensor C of shape [..., m, p]. The operation leverages multithreading to optimize performance.
     *
     * @param other The tensor B to multiply with the current tensor A.
     * @param n_threads Number of threads to utilize for parallel computation.
     * @return Tensor Resulting tensor C of shape [..., m, p].
     */
    Tensor matmul(const Tensor &other) const;

    /**
     * @brief Computes the sum of the tensor elements along a given dimension.
     *
     * This function reduces the tensor by summing over the specified dimension,
     * producing a new tensor with one less dimension.
     * 
     * It supports **automatic differentiation**, meaning the sum operation 
     * contributes to backpropagation.
     *
     * @param dim The dimension along which to compute the sum.
     * @return Tensor A reduced tensor with one less dimension.
     *
     * @throws std::invalid_argument If `dim` is out of bounds.
     *
     * @note The resulting tensor retains `requires_grad` if the original tensor does.
     *
     * @example
     * @code
     * Tensor t({{1.0, 2.0}, {3.0, 4.0}}, {2, 2}, true); // Shape: (2,2)
     * Tensor s = t.sum(0); // Shape: (1,2), values: {4.0, 6.0}
     * @endcode
     */
    Tensor sum(const size_t dim) const;

    /**
     * @brief Computes the sum over the last dimension of the tensor.
     *
     * This is a convenience function that calls `sum(dim)`, where `dim`
     * is the last axis (`shape.size() - 1`). It reduces the tensor by summing
     * along the trailing dimension.
     *
     * @return Tensor A reduced tensor with one less dimension.
     *
     * @throws std::invalid_argument If the tensor has no dimensions.
     *
     * @note Supports **automatic differentiation** and retains `requires_grad` if applicable.
     *
     * @example
     * @code
     * Tensor t({{1.0, 2.0}, {3.0, 4.0}}, {2, 2}, true);
     * Tensor s = t.sum(); // Shape: (2,), values: {3.0, 7.0}
     * @endcode
     */
    Tensor sum() const;

    /**
     * @brief Computes the mean of the tensor elements along a given dimension.
     *
     * This function reduces the tensor by computing the average over the specified dimension,
     * producing a new tensor with one less dimension.
     *
     * It supports **automatic differentiation**, allowing gradients to propagate correctly.
     *
     * @param dim The dimension along which to compute the mean.
     * @return Tensor A reduced tensor with one less dimension.
     *
     * @throws std::invalid_argument If `dim` is out of bounds.
     *
     * @note The resulting tensor retains `requires_grad` if the original tensor does.
     *
     * @example
     * @code
     * Tensor t({{1.0, 2.0}, {3.0, 4.0}}, {2, 2}, true); // Shape: (2,2)
     * Tensor m = t.mean(0); // Shape: (1,2), values: {2.0, 3.0}
     * @endcode
     */
    Tensor mean(const size_t dim) const;
    
    /**
     * @brief Computes the mean over the last dimension of the tensor.
     *
     * This is a convenience function that calls `mean(dim)`, where `dim`
     * is the last axis (`shape.size() - 1`). It reduces the tensor by averaging
     * along the trailing dimension.
     *
     * @return Tensor A reduced tensor with one less dimension.
     *
     * @throws std::invalid_argument If the tensor has no dimensions.
     *
     * @note Supports **automatic differentiation** and retains `requires_grad` if applicable.
     *
     * @example
     * @code
     * Tensor t({{1.0, 2.0}, {3.0, 4.0}}, {2, 2}, true);
     * Tensor m = t.mean(); // Shape: (2,), values: {1.5, 3.5}
     * @endcode
     */
    Tensor mean() const;

    /**
     * @brief Computes the element-wise natural logarithm of the tensor.
     *
     * This function applies the natural logarithm (\f$\log(x)\f$) to each element
     * of the tensor. The operation supports **automatic differentiation**, meaning
     * gradients are computed correctly during backpropagation.
     *
     * **Gradient computation:**  
     * If `y = x.log()`, then during backpropagation:
     * \f[
     * \frac{dL}{dx} = \frac{dL}{dy} \cdot \frac{1}{x}
     * \f]
     *
     * @return Tensor A tensor where each element is the natural logarithm of the original tensor.
     *
     * @throws std::domain_error If any element in the tensor is non-positive (as \f$\log(x)\f$
     *         is undefined for \f$x \leq 0\f$).
     *
     * @note The resulting tensor retains `requires_grad` if the original tensor does.
     *
     * @example
     * @code
     * Tensor t({1.0, 2.0, 3.0}, {3}, true); // Shape: (3,)
     * Tensor log_t = t.log();               // Shape: (3,), values: {0.0, 0.693, 1.098}
     * @endcode
     */
    Tensor log() const;
   
    /**
     * @brief Computes the element-wise exponential of the tensor.
     *
     * This function applies the exponential function (\f$e^x\f$) to each element
     * of the tensor. The operation supports **automatic differentiation**, meaning
     * gradients are computed correctly during backpropagation.
     *
     * **Gradient computation:**  
     * If `y = x.exp()`, then during backpropagation:
     * \f[
     * \frac{dL}{dx} = \frac{dL}{dy} \cdot e^x
     * \f]
     *
     * @return Tensor A tensor where each element is the exponential of the original tensor.
     *
     * @note The resulting tensor retains `requires_grad` if the original tensor does.
     *
     * @example
     * @code
     * Tensor t({0.0, 1.0, 2.0}, {3}, true); // Shape: (3,)
     * Tensor exp_t = t.exp();               // Shape: (3,), values: {1.0, 2.718, 7.389}
     * @endcode
     */
    Tensor exp() const; 
   
    /**
     * @brief Computes the softmax function along a specified dimension.
     *
     * The softmax function normalizes the input tensor along the given dimension,
     * ensuring that the output values sum to 1 along that axis. It is defined as:
     * \f[
     * \text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
     * \f]
     * 
     * A small constant (\f$10^{-6}\f$) is added to the denominator for numerical stability.
     *
     * @param dim The dimension along which to apply the softmax function.
     * @return Tensor The softmax-normalized tensor.
     *
     * @example
     * @code
     * Tensor logits = Tensor({2.0, 1.0, 0.1}, {3});
     * Tensor probabilities = logits.softmax(0);
     * @endcode
     */
    Tensor softmax(size_t dim) const;
    
    /**
     * @brief Computes the softmax function along the last dimension.
     *
     * This function applies the softmax operation along the last axis of the tensor.
     * It is equivalent to calling `softmax(dim)` with `dim = tensor.ndim() - 1`.
     *
     * @return Tensor The softmax-normalized tensor.
     *
     * @example
     * @code
     * Tensor logits = Tensor({{2.0, 1.0, 0.1}, {0.5, 0.7, 0.2}}, {2, 3});
     * Tensor probabilities = logits.softmax();
     * @endcode
     */
    Tensor softmax() const;
    
    /**
     * @brief Converts a tensor of class indices into a one-hot encoded representation.
     *
     * This function expands the input tensor along a new trailing dimension, representing
     * class indices as one-hot vectors. The resulting tensor has shape `[..., num_classes]`,
     * where each value is replaced with a one-hot encoded vector.
     *
     * @param num_classes The total number of classes in the encoding.
     * @return Tensor The one-hot encoded tensor.
     * @throws std::invalid_argument If any input value is out of bounds for `num_classes`.
     *
     * @example
     * @code
     * Tensor labels = Tensor({0, 2, 1}, {3});
     * Tensor one_hot = labels.onehot_encode(3);
     * @endcode
     */
    Tensor onehot_encode(size_t num_classes) const;
    /* activation function */

    /**
     * @brief Applies the ReLU (Rectified Linear Unit) activation function element-wise.
     *
     * This function computes the ReLU activation for each element in the tensor,
     * defined as:
     * \f[
     * f(x) = \max(0, x)
     * \f]
     *
     * If the input tensor has `requires_grad` set to `true`, this function:
     * - Constructs a computation graph entry for automatic differentiation.
     * - Stores a backward function that computes the derivative of ReLU, which is
     *   `1` for positive inputs and `0` for non-positive inputs.
     * - Ensures per-thread gradient tracking for multi-threaded execution.
     *
     * @return Tensor The result tensor with the same shape as the input tensor.
     * 
     * @note If `requires_grad` is `true`, this function registers the tensor in the computation graph
     *       and initializes thread-local gradients.
     * 
     * @example Usage:
     * @code
     * Tensor x({-2.0f, 3.0f, -1.0f, 4.0f}, {2, 2}, true);
     * Tensor y = x.relu();
     * @endcode
     */
    Tensor relu() const;

    /* initialization functions: */

    /**
     * @brief Creates a tensor with elements sampled from a uniform distribution U(0, 1).
     *
     * This function generates a tensor of the given shape where each element is drawn
     * independently from a uniform distribution in the range \f$[0, 1]\f$.
     *
     * @param shape The desired shape of the tensor.
     * @param requires_grad If `true`, enables gradient computation for this tensor.
     * @return Tensor A tensor initialized with random values sampled from \f$U(0, 1)\f$.
     *
     * @note Uses a global random number generator (`global_generator`).
     *
     * @example
     * @code
     * Tensor t = Tensor::randn({3, 3}, true); // 3x3 random tensor, requires gradient
     * @endcode
     */
    static Tensor randn(const std::vector<size_t>& shape, bool requires_grad = false);

    /**
     * @brief Creates a tensor using He initialization.
     *
     * He initialization is designed for layers with ReLU activations to improve weight scaling.
     * The elements are sampled from a uniform distribution scaled by a standard deviation
     * of \f$\sqrt{\frac{2}{\text{in_features}}}\f$.
     *
     * @param in_features Number of input features (fan-in).
     * @param out_features Number of output features.
     * @param requires_grad If `true`, enables gradient computation for this tensor.
     * @return Tensor A tensor of shape `(in_features, out_features)` initialized using He initialization.
     *
     * @note Uses a global random number generator (`global_generator`).
     *
     * @example
     * @code
     * Tensor weights = Tensor::randn_he(128, 64, true); // He-initialized 128x64 weight matrix
     * @endcode
     */
    static Tensor randn_he(size_t in_features, size_t out_features, bool requires_grad);
    
    /**
     * @brief Creates a bias tensor with uniform initialization following PyTorch's convention.
     *
     * The bias values are sampled uniformly from the range \f$[-b, b]\f$, where
     * \f$b = \frac{1}{\sqrt{\text{in_features}}}\f$. This method follows PyTorch's
     * default bias initialization.
     *
     * @param in_features Number of input features (fan-in).
     * @param requires_grad If `true`, enables gradient computation for this tensor.
     * @return Tensor A tensor of shape `(in_features,)` initialized for use as a bias vector.
     *
     * @note Uses a global random number generator (`global_generator`).
     *
     * @example
     * @code
     * Tensor bias = Tensor::bias_uniform(128, true); // Bias tensor for 128 input features
     * @endcode
     */
    static Tensor bias_uniform(size_t in_features, bool requires_grad);


    /* helper functions: */

    /**
     * @brief Resets the gradient of the tensor to zero.
     *
     * This function is used to clear accumulated gradients before a new backward pass.
     */
    void zero_grad(); 
    
    /**
     * @brief Recursively prints tensor data in a structured format.
     *
     * This function is used internally by the `<<` operator to print tensor values
     * in a **multi-dimensional array format**. It ensures that nested dimensions
     * are displayed correctly.
     *
     * @param os The output stream to write to.
     * @param dim The current dimension being processed.
     * @param offset The starting index in the tensor's data vector.
     * @param stride The step size to traverse along the current dimension.
     *
     * @example
     * @code
     * Tensor t({1.0, 2.0, 3.0, 4.0}, {2, 2});
     * std::cout << t; // Calls print_recursive() internally
     * @endcode
     */
    void print_recursive(std::ostream& os, size_t dim, size_t offset, size_t stride) const; 
    
    /**
     * @brief Returns the shape of the tensor.
     *
     * @return std::vector<size_t> The shape vector of the tensor.
     */
    std::vector<size_t> shape() const;

    /**
     * @brief Returns the raw data of the tensor as a vector.
     *
     * @return std::vector<float> The data of the tensor.
     */
    std::vector<float> data() const;

    /**
     * @brief Retrieves the gradient of the tensor.
     *
     * @return Tensor A tensor containing the gradient.
     * @throws std::runtime_error If the tensor has no gradient.
     */
    Tensor grad() const;

    /**
     * @brief Disables gradient computation for the tensor.
     *
     * This is typically used in inference mode to improve efficiency.
     */
    void eval();

    /**
     * @brief Enables gradient computation for the tensor.
     *
     * This is required when training a model.
     */
    void train();

    private:
};

/**
 * @brief Computes the Cross Entropy Loss between predicted and true labels.
 *
 * This function computes the categorical cross-entropy loss.
 * It first applies softmax to `y_pred` along the last dimension, 
 * then computes the negative log-likelihood weighted by one-hot
 * encoded `y_true`. Finally, it returns the mean loss over the batch dimension.
 *
 * The formula for cross-entropy loss is:
 * \f[
 * L = - \frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{C} y_{ij} \log(\text{softmax}(x_{ij}))
 * \f]
 *
 * @param y_pred Tensor of shape `(batch_size, num_classes)`, representing model logits.
 * @param y_true Tensor of shape `(batch_size)`, containing ground truth class indices.
 * @return Tensor The scalar cross-entropy loss.
 *
 * @example
 * @code
 * Tensor logits = Tensor({{2.0, 1.0, 0.1}, {0.5, 0.7, 0.2}}, {2, 3});
 * Tensor labels = Tensor({0, 2}, {2});
 * Tensor loss = CrossEntropyLoss(logits, labels);
 * @endcode
 */
Tensor CrossEntropyLoss(const Tensor& y_pred, const Tensor& y_true);
#endif // CPPGRAD_H