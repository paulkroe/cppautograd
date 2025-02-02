### CPP Autograd
CPP autograd is a simple autograd engine written in C++.
Its syntax aims to mirror LibTorch.

For some examples please see `grad/demo/` or the Tutorial section below.

## Optimization
Removed usage of std::vector<float>::operator[](unsigned long) from matrix multiplication.
Indexing into matrices, especailly during matmul is inefficient and using operator[] hurts cache performance. I.e. replace:
```
for (size_t i = 0; i < N; i++) {
    result[i] = a[i] + b[i];
}
```
by
```
float* res_ptr = result.data();
const float* a_ptr = a.data();
const float* b_ptr = b.data();

for (size_t i = 0; i < N; i++) {
    *res_ptr++ = *a_ptr++ + *b_ptr++;
}
```
Fetching gradients instead of calling grad() when updating gradients.  (calling grad took as much as 5% of the overall training time)

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
