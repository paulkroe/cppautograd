#include <torch/torch.h>
#include "../vanilla/cppgrad.h"
#include <iostream>
#include <cmath>

// Function to compare scalar tensors
void compare_scalars(float cpp_value, float torch_value, const std::string& test_name) {
    if (std::abs(cpp_value - torch_value) < 1e-6) {
        std::cout << test_name << " PASSED.\n";
    } else {
        std::cerr << test_name << " FAILED: " << cpp_value << " (cpp) vs " << torch_value << " (torch).\n";
    }
}

// Function to test scalar operations
void test_scalars() {
    // Test 1: Scalar addition
    {
        Tensor a_cpp({1.0}, true);
        Tensor b_cpp({2.0}, true);

        Tensor c_cpp = a_cpp + b_cpp;
        c_cpp.backward();

        auto a_torch = torch::tensor({1.0}, torch::requires_grad());
        auto b_torch = torch::tensor({2.0}, torch::requires_grad());

        auto c_torch = a_torch + b_torch;
        c_torch.backward();

        // Compare results
        compare_scalars(c_cpp.data[0], c_torch.item<float>(), "Addition Result");
        compare_scalars(a_cpp.grad->data[0], a_torch.grad().item<float>(), "Addition Gradient (a)");
        compare_scalars(b_cpp.grad->data[0], b_torch.grad().item<float>(), "Addition Gradient (b)");
    }

    // Test 2: Scalar multiplication
    {
        Tensor a_cpp({3.0}, true);
        Tensor b_cpp({4.0}, true);

        Tensor c_cpp = a_cpp * b_cpp;
        c_cpp.backward();

        auto a_torch = torch::tensor({3.0}, torch::requires_grad());
        auto b_torch = torch::tensor({4.0}, torch::requires_grad());

        auto c_torch = a_torch * b_torch;
        c_torch.backward();

        // Compare results
        compare_scalars(c_cpp.data[0], c_torch.item<float>(), "Multiplication Result");
        compare_scalars(a_cpp.grad->data[0], a_torch.grad().item<float>(), "Multiplication Gradient (a)");
        compare_scalars(b_cpp.grad->data[0], b_torch.grad().item<float>(), "Multiplication Gradient (b)");
    }

    // Test 3: Mean reduction (scalar)
    {
        Tensor a_cpp({2.0}, true);

        Tensor c_cpp = a_cpp.mean();
        c_cpp.backward();

        auto a_torch = torch::tensor({2.0}, torch::requires_grad());

        auto c_torch = a_torch.mean();
        c_torch.backward();

        // Compare results
        compare_scalars(c_cpp.data[0], c_torch.item<float>(), "Mean Result");
        compare_scalars(a_cpp.grad->data[0], a_torch.grad().item<float>(), "Mean Gradient");
    }
}

// Function to compare multidimensional tensors
void compare_tensors(const std::vector<float>& cpp_data, const torch::Tensor& torch_tensor, const std::string& test_name) {
    auto torch_data = torch_tensor.flatten().data_ptr<float>();
    for (size_t i = 0; i < cpp_data.size(); ++i) {
        if (std::abs(cpp_data[i] - torch_data[i]) > 1e-6) {
            std::cerr << test_name << " FAILED at index " << i << ": " << cpp_data[i] << " (cpp) vs " << torch_data[i] << " (torch).\n";
            return;
        }
    }
    std::cout << test_name << " PASSED.\n";
}

// Function to test multidimensional tensor operations
void test_multidimensional() {
    // Test 1: Element-wise addition
    {
        std::cout << "Test 1: Element-wise addition\n";
        Tensor a_cpp({1.0, 2.0, 3.0}, {3}, true);
        Tensor b_cpp({4.0, 5.0, 6.0}, {3}, true);

        Tensor c_cpp = (a_cpp + b_cpp).sum();

        c_cpp.backward();

        auto a_torch = torch::tensor({1.0, 2.0, 3.0}, torch::requires_grad());
        auto b_torch = torch::tensor({4.0, 5.0, 6.0}, torch::requires_grad());

        auto c_torch = (a_torch + b_torch).sum(); // Reduce for backward
        c_torch.backward();

        compare_tensors(c_cpp.data, c_torch, "Addition Result");
        compare_tensors(a_cpp.grad->data, a_torch.grad(), "Addition Gradient (a)");
        compare_tensors(b_cpp.grad->data, b_torch.grad(), "Addition Gradient (b)");
    }

    // Test 2: Element-wise multiplication
    {
        std::cout << "Test 2: Element-wise multiplication\n";
        Tensor a_cpp({1.0, 2.0, 3.0}, {3}, true);
        Tensor b_cpp({4.0, 5.0, 6.0}, {3}, true);

        Tensor c_cpp = (a_cpp * b_cpp).mean();

        c_cpp.backward();

        auto a_torch = torch::tensor({1.0, 2.0, 3.0}, torch::requires_grad());
        auto b_torch = torch::tensor({4.0, 5.0, 6.0}, torch::requires_grad());

        auto c_torch = (a_torch * b_torch).mean(); // Reduce for backward
        c_torch.backward();

        compare_tensors(c_cpp.data, c_torch, "Multiplication Result");
        compare_tensors(a_cpp.grad->data, a_torch.grad(), "Multiplication Gradient (a)");
        compare_tensors(b_cpp.grad->data, b_torch.grad(), "Multiplication Gradient (b)");
    }

    // Test 3: Matrix multiplication
    {
        std::cout << "Test 3: Matrix Vector multiplication\n";
        Tensor a_cpp({1.0, 2.0, 3.0, 4.0}, {2, 2}, true);
        Tensor b_cpp({7.0, 8.0}, {2, 1}, true);
        
        Tensor c_cpp = a_cpp.matmul(b_cpp);
        c_cpp.shape = {2};
        c_cpp = c_cpp.sum();
        c_cpp.backward();

        auto a_torch = torch::tensor({{1.0, 2.0}, {3.0, 4.0}}, torch::requires_grad());
        auto b_torch = torch::tensor({{7.0}, {8.0}}, torch::requires_grad());
        
        auto c_torch = torch::matmul(a_torch, b_torch).sum();
        c_torch.backward();

        compare_tensors(c_cpp.data, c_torch, "Matrix Vector Multiplication Result");
        compare_tensors(a_cpp.grad->data, a_torch.grad(), "Matrix Vector Gradient (a)");
        compare_tensors(b_cpp.grad->data, b_torch.grad(), "Matrix Vector Gradient (b)");
    }
    // Test 4: Matrix multiplication
    {
        std::cout << "Test 4: Matrix multiplication\n";
        Tensor a_cpp({1.0, 2.0, 3.0, 4.0}, {2, 2}, true);
        Tensor b_cpp({5.0, 6.0, 7.0, 8.0}, {2, 2}, true);
        Tensor c_cpp({7.0, 8.0}, {2, 1}, true);

        Tensor d_cpp = ((a_cpp.matmul(b_cpp)).matmul(c_cpp));
        d_cpp.shape = {2};
        d_cpp = d_cpp.sum();
        d_cpp.backward();
        auto a_torch = torch::tensor({{1.0, 2.0}, {3.0, 4.0}}, torch::requires_grad());
        auto b_torch = torch::tensor({{5.0, 6.0}, {7.0, 8.0}}, torch::requires_grad());
        auto c_torch = torch::tensor({{7.0}, {8.0}}, torch::requires_grad());

        auto d_torch = torch::matmul(torch::matmul(a_torch, b_torch), c_torch).sum();
        d_torch.backward();

        compare_tensors(d_cpp.data, d_torch, "Matrix Multiplication Result");
        compare_tensors(a_cpp.grad->data, a_torch.grad(), "Matrix Multiplication Gradient (a)");
        compare_tensors(b_cpp.grad->data, b_torch.grad(), "Matrix Multiplication Gradient (b)");
    }

    // Test 5: Mean reduction
    {
        Tensor a_cpp({1.0, 2.0}, {2}, true);

        Tensor c_cpp = a_cpp.mean();
        c_cpp.backward();

        auto a_torch = torch::tensor({{1.0, 2.0}}, torch::requires_grad());
        auto c_torch = a_torch.mean();
        c_torch.backward();

        compare_tensors(c_cpp.data, c_torch, "Mean Reduction Result");
        compare_tensors(a_cpp.grad->data, a_torch.grad(), "Mean Reduction Gradient");
    }
}

int main() {
    std::cout << "Testing scalar operations...\n";
    test_scalars();

    std::cout << "\nTesting multidimensional operations...\n";
    test_multidimensional();

    return 0;
}
