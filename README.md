### CPP Autograd
CPP autograd is a simple autograd engine written in C++.
Its syntax aims to mirror LibTorch.

For some examples please see `grad/demo/` or the Tutorial section below.

## Speedup
speedup using multiple cpus:
two areas of speedup:
1) do the matrix multiplication on multiple cores
2) distributed training

WIP: parallelization on multiple cores, adding a CUDA kernel

plan of action:
first make matmul more efficient than just the most naive implementation
then make matmul backprop more efficient as well
add support for multiple threads to backward
(Strassen algorithm)

then add multithreadding support to other operators as well (not sure if they are beneficial)

then measure duration of training both with and without acceleration

after doing that, split training into batches and experiement what is the fastest

## Tutorial

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

## Usage

## Design Choices
- Make no difference when broadcasting
- Tensors having unique IDs
- using make_shared(*this) copies data instead of creating a ref
- Backpropagation should be performed using topological sorting
