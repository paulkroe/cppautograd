/*
This tutorial demonstrates how to use the autograd engine step by step.
We will explore scalar operations, multi-dimensional tensors, and backpropagation.
*/

#include <torch/torch.h>
#include "../grad/cppgrad.h"
#include <iostream>

int main() {

    std::cout << "=== Scalar Operations ===\n";
    // Step 1: Simple scalar operations
    // Define two scalar tensors with values 3.0 and 4.0
    Tensor a = Tensor({3.0}, true);
    Tensor b = Tensor({4.0}, true);
    
    // Perform multiplication and addition: c = a * b + 2
    Tensor c = a * b + 2.0;
    std::cout << "c = a * b + 2 -> " << c << "\n";

    std::cout << "=== Creating Multi-Dimensional Tensors ===\n";
    // Step 2: Creating multi-dimensional tensors
    // Define a 2x3 tensor with explicit values
    Tensor d = Tensor({1,2,3,4,5,6}, {2, 3}, true);
    
    // Create a random 2x3 tensor with autograd enabled
    Tensor e = Tensor::randn({2, 3}, true);
    std::cout << "Random tensor d: " << d << "\n";

    std::cout << "=== Sum and Mean Operations ===\n";
    // Step 3: Demonstrating sum and mean operations
    // Compute the sum of all elements in d
    Tensor f = d.sum();
    // Compute the mean of all elements in e
    Tensor g = e.mean();
    std::cout << "Sum of d: " << f << "\n";
    std::cout << "Mean of e: " << g << "\n";

    std::cout << "=== Reducing to Scalar for Backpropagation ===\n";
    // Step 4: Reducing a multi-dimensional tensor to a scalar before backpropagation
    // Perform element-wise multiplication, then sum twice to ensure scalar output
    Tensor h = (d * e).sum().sum();
    h.backward(); // Compute gradients
    
    // Print the computed gradients
    std::cout << "Gradient of d: " << d.grad() << "\n";
    std::cout << "Gradient of e: " << e.grad() << "\n";

    std::cout << "=== Backpropagation ===\n";
    // Step 5: Computing gradients with a mathematical function
    // Create a random 3x3 tensor
    Tensor x = Tensor::randn({3, 3}, true);
    
    // Apply element-wise exponential and logarithm, then compute mean and sum
    Tensor y = (x.exp() + x.log()).mean().sum();
    y.backward(); // Compute gradients
    
    std::cout << "Gradient of x: " << x.grad() << "\n";

    std::cout << "=== Multi-Dimensional Tensors ===\n";
    // Step 6: Performing more complex tensor operations
    // Creating three 4x4 random tensors
    Tensor m1 = Tensor::randn({4, 4}, true);
    Tensor m2 = Tensor::randn({4, 4}, true);
    Tensor m3 = Tensor::randn({4, 4}, true);
    
    // Compute result using element-wise multiplication and addition
    Tensor result = m1 * m2 + m3;
    result.sum().sum().backward(); // Reduce to scalar and compute gradients
    
    std::cout << "Gradient of m1: " << m1.grad() << "\n";

    std::cout << "=== Broadcasting Capabilities ===\n";
    // Step 7: Demonstrating broadcasting capabilities
    // Define a scalar tensor and a larger 2x3 tensor
    Tensor small = Tensor({3.0}, true);
    Tensor large = Tensor::randn({2, 3}, true);
    
    // Broadcasting: small is expanded to match the shape of large
    Tensor broadcasted = small * large;
    std::cout << "Broadcasted multiplication: " << broadcasted << "\n";
    
    // Reduce to scalar before backpropagation
    broadcasted.sum().sum().backward();
    std::cout << "Gradient of small: " << small.grad() << "\n";

    return 0;
}
