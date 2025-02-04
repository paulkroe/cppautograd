# CPP Autograd
CPP autograd is a simple autograd engine written in C++.
Its syntax aims to mirror LibTorch.

For some examples please see `demo/` or the Tutorial section below.

## Performance Optimization

### Benchmarking Performance

To assess the performance of my implementation, I trained a simple five-layer linear model on the MNIST digit classification task. The goal was not to achieve high accuracy but rather to measure training speed.

Initially, I trained the model for a few batches (batch size = 32). To my surprise, each batch took approximately one minute, meaning a full training run would take around 30 hours. Profiling the code using gprof revealed that during just a few batches, the program made 121 million calls to std::vector<float>::operator[](unsigned long), the standard library function for accessing vector elements. This function ensures safe access by verifying index validity, but its frequent invocation caused significant overhead.

To address this, I replaced the [] operator with direct pointer arithmetic, optimizing matrix operations from something like:
```cpp
for (size_t i = 0; i < N; i++) {
    result[i] = a[i] + b[i];
}
```
to:
```cpp
float* res_ptr = result.data();
const float* a_ptr = a.data();
const float* b_ptr = b.data();

for (size_t i = 0; i < N; i++) {
    *res_ptr++ = *a_ptr++ + *b_ptr++;
}
```

This reduced the time per batch from 60 seconds to 15 seconds.

Multithreading

To further improve performance, I implemented multithreaded tiled matrix multiplication. However, the results were underwhelming—using between 2 and 8 cores only yielded a 10% speedup. I identified two primary bottlenecks:

Thread Overhead: The largest matrix involved was 728×512, making the cost of creating and managing threads disproportionately high.

Memory Bandwidth Limitations: The processor was too fast for its memory bandwidth, meaning the bottleneck was not computation but rather memory access—an issue commonly referred to as the Von Neumann bottleneck.

### Addressing Thread Overhead

To amortize thread creation costs, I parallelized training at the batch level instead of individual matrix operations. Instead of processing a single batch at a time, I let n threads process n batches simultaneously, then synchronized gradients by averaging them across all threads before updating model weights. This is equivalent to increasing the batch size, which often stabilizes training and improves convergence.

On a technical level, each tensor retained a separate gradient per thread, allowing shared model weights without copying them across threads.

Unfortunately, this approach increased batch time instead of decreasing it—doubling the time per batch. Profiling confirmed that memory allocation was the primary issue.

Addressing Memory Bottlenecks

The memory issue stemmed from how the backward pass was implemented. Consider the following example:

```cpp
Tensor a = (Tensor({1,2,3,4}, {4}) + Tensor({1,2,3,4}, {4})).sum();
```

This creates an intermediate tensor for the addition result. To compute gradients, backpropagation requires access to this intermediate tensor. The naive solution was to store deep copies of intermediate results, ensuring availability during backpropagation. However, this led to excessive memory usage—every simple addition created three tensors, two of which were redundant copies.

To fix this, I introduced a wrapper class around tensor data. This allowed efficient memory management by ensuring that intermediate tensors were deallocated as soon as they were no longer needed. This solved the issue and allowed for efficient parallel execution.

### Results

Fixing memory issues provided the expected performance gains. The final improvements were:

Optimized indexing: Replacing std::vector::operator[] with pointer arithmetic significantly improved cache performance.

Efficient memory management: Reducing redundant tensor allocations drastically decreased memory overhead.

The average batch time dropped from 60 seconds to just 0.13 seconds.

While this is not a comprehensive benchmark, the optimizations successfully removed major bottlenecks, making training vastly more efficient.

## Usage

This tutorial demonstrates how to use the autograd engine step by step.
We will explore scalar operations, multi-dimensional tensors, and backpropagation. The code below can also be found in `grad/demo/tutorial.cpp`.

### Scalar Operations
We begin by defining simple scalar tensors and performing basic operations such as multiplication and addition.
```cpp
Tensor a = Tensor({3.0}, true);
Tensor b = Tensor({4.0}, true);
Tensor c = a * b + 2.0;
std::cout << "c = a * b + 2 -> " << c << "\n";
```

### Creating Multi-Dimensional Tensors
We can create multi-dimensional tensors explicitly or generate random ones.
```cpp
Tensor d = Tensor({1,2,3,4,5,6}, {2, 3}, true);
Tensor e = Tensor::randn({2, 3}, true);
std::cout << "Random tensor d: " << d << "\n";
```

### Sum and Mean Operations
Summing and averaging elements across tensors is straightforward.
```cpp
Tensor f = d.sum();
Tensor g = e.mean();
std::cout << "Sum of d: " << f << "\n";
std::cout << "Mean of e: " << g << "\n";
```

### Reducing to Scalar for Backpropagation
Before computing gradients, we need to reduce tensors to scalars.
```cpp
Tensor h = (d * e).sum().sum();
h.backward();
std::cout << "Gradient of d: " << d.get_grad() << "\n";
std::cout << "Gradient of e: " << e.get_grad() << "\n";
```

### Backpropagation
Autograd supports differentiating complex operations.
```cpp
Tensor x = Tensor::randn({3, 3}, true);
Tensor y = (x.exp() + x.log()).mean().sum();
y.backward();
std::cout << "Gradient of x: " << x.get_grad() << "\n";
```

### Multi-Dimensional Tensors
Operations involving multiple tensors work seamlessly with autograd.
```cpp
Tensor m1 = Tensor::randn({4, 4}, true);
Tensor m2 = Tensor::randn({4, 4}, true);
Tensor m3 = Tensor::randn({4, 4}, true);
Tensor result = m1 * m2 + m3;
result.sum().sum().backward();
std::cout << "Gradient of m1: " << m1.get_grad() << "\n";
```

### Broadcasting Capabilities
Broadcasting allows operations between tensors of different shapes.
```cpp
Tensor small = Tensor({3.0}, true);
Tensor large = Tensor::randn({2, 3}, true);
Tensor broadcasted = small * large;
std::cout << "Broadcasted multiplication: " << broadcasted << "\n";
broadcasted.sum().sum().backward();
std::cout << "Gradient of small: " << small.get_grad() << "\n";
```