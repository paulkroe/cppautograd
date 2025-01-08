#include <torch/torch.h>
#include "cppgrad.h"
#include <iostream>
#include <cmath>

void compare_tensors(const std::vector<float>& cpp_data, const torch::Tensor& torch_tensor, const std::string& label) {
    auto torch_data = torch_tensor.flatten().data_ptr<float>();
    bool match = true;
    for (size_t i = 0; i < cpp_data.size(); ++i) {
        if (std::abs(cpp_data[i] - torch_data[i]) > 1e-6) {
            match = false;
            break;
        }
    }
    if (match) {
        std::cout << label << " match!" << std::endl;
    } else {
        std::cout << label << " mismatch!" << std::endl;
    }
}

int main() {
    // Test 1: Matrix Multiplication
    {
        // Initialize LibTorch tensors
        auto a = torch::tensor({{1.0, 2.0}, {1.0, 2.0}, {1.0, 2.0}}, torch::requires_grad());
        auto b = torch::tensor({{4.0, 1.0}, {4.0, 1.0}}, torch::requires_grad());

        // Perform matrix multiplication with LibTorch
        auto d_torch = torch::matmul(a, b).sum(); // Reduce to scalar for backward
        d_torch.backward();

        // Perform the same operations in cppgrad
        Tensor a_cpp({1.0, 2.0, 1.0, 2.0, 1.0, 2.0}, {3, 2}, true);
        Tensor b_cpp({4.0, 1.0, 4.0, 1.0}, {2, 1}, true);
        Tensor d_cpp = a_cpp.matmul(b_cpp).sum(); // Reduce to scalar for backward
        d_cpp.backward();

        // Compare results
        compare_tensors(d_cpp.data, d_torch, "Result");
        compare_tensors(a_cpp.grad->data, a.grad(), "Gradient of a");
        compare_tensors(b_cpp.grad->data, b.grad(), "Gradient of b");
    }

    // Test 2: Element-wise Addition
    {
        // Initialize LibTorch tensors
        auto a = torch::tensor({1.0, 2.0, 3.0}, torch::requires_grad());
        auto b = torch::tensor({4.0, 5.0, 6.0}, torch::requires_grad());

        // Perform addition with LibTorch
        auto d_torch = (a + b).sum(); // Reduce to scalar for backward
        d_torch.backward();

        // Perform the same operations in cppgrad
        Tensor a_cpp({1.0, 2.0, 3.0}, true);
        Tensor b_cpp({4.0, 5.0, 6.0}, true);
        Tensor d_cpp = (a_cpp + b_cpp).sum(); // Reduce to scalar for backward
        d_cpp.backward();

        // Compare results
        compare_tensors(d_cpp.data, d_torch, "Result");
        compare_tensors(a_cpp.grad->data, a.grad(), "Gradient of a");
        compare_tensors(b_cpp.grad->data, b.grad(), "Gradient of b");
    }

    // Test 3: Element-wise Multiplication
    {
        // Initialize LibTorch tensors
        auto a = torch::tensor({1.0, 2.0, 3.0}, torch::requires_grad());
        auto b = torch::tensor({4.0, 5.0, 6.0}, torch::requires_grad());

        // Perform multiplication with LibTorch
        auto d_torch = (a * b).sum(); // Reduce to scalar for backward
        d_torch.backward();

        // Perform the same operations in cppgrad
        Tensor a_cpp({1.0, 2.0, 3.0}, true);
        Tensor b_cpp({4.0, 5.0, 6.0}, true);
        Tensor d_cpp = (a_cpp * b_cpp).sum(); // Reduce to scalar for backward
        d_cpp.backward();

        // Compare results
        compare_tensors(d_cpp.data, d_torch, "Result");
        compare_tensors(a_cpp.grad->data, a.grad(), "Gradient of a");
        compare_tensors(b_cpp.grad->data, b.grad(), "Gradient of b");
    }

    // Test 5: Mean Reduction
    {
        // Initialize LibTorch tensors
        auto a = torch::tensor({1.0, 2.0, 3.0}, torch::requires_grad());

        // Perform mean reduction with LibTorch
        auto d_torch = a.mean(); // Scalar output
        d_torch.backward();

        // Perform the same operations in cppgrad
        Tensor a_cpp({1.0, 2.0, 3.0}, true);
        Tensor d_cpp = a_cpp.mean(); // Scalar output
        d_cpp.backward();

        // Compare results
        compare_tensors(d_cpp.data, d_torch, "Result");
        compare_tensors(a_cpp.grad->data, a.grad(), "Gradient of a");
    }

    return 0;
}