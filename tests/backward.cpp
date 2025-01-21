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
        if (std::abs(cpp_data[i] - torch_data[i]) > 0.1) {
            std::cerr << test_name << " FAILED at index " << i << ": " << cpp_data[i] << " (cpp) vs " << torch_data[i] << " (torch).\n";
            std::cerr << "Difference: " << std::abs(cpp_data[i] - torch_data[i]) << std::endl;
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

    // Test 3: Element-wise division
    {
        std::cout << "Test 3: Element-wise division\n";
        Tensor a_cpp({1.0, 2.0, 3.0}, {3}, true);
        Tensor b_cpp({4.0, 5.0, 6.0}, {3}, true);

        Tensor c_cpp = (a_cpp / b_cpp).mean();

        c_cpp.backward();

        auto a_torch = torch::tensor({1.0, 2.0, 3.0}, torch::requires_grad());
        auto b_torch = torch::tensor({4.0, 5.0, 6.0}, torch::requires_grad());

        auto c_torch = (a_torch / b_torch).mean(); // Reduce for backward
        c_torch.backward();

        compare_tensors(c_cpp.data, c_torch, "Division Result");
        compare_tensors(a_cpp.grad->data, a_torch.grad(), "Division Gradient (a)");
        compare_tensors(b_cpp.grad->data, b_torch.grad(), "Division Gradient (b)");
    }

    // Test 4: Matrix multiplication
    {
        std::cout << "Test 4: Matrix Vector multiplication\n";
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
    // Test 5: Matrix multiplication
    {
        std::cout << "Test 5: Matrix multiplication\n";
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

    // Test 6: Mean reduction
    {
        std::cout << "Test 6: Mean reduction\n";
        Tensor a_cpp({1.0, 2.0}, {2}, true);

        Tensor c_cpp = a_cpp.mean();
        c_cpp.backward();

        auto a_torch = torch::tensor({{1.0, 2.0}}, torch::requires_grad());
        auto c_torch = a_torch.mean();
        c_torch.backward();

        compare_tensors(c_cpp.data, c_torch, "Mean Reduction Result");
        compare_tensors(a_cpp.grad->data, a_torch.grad(), "Mean Reduction Gradient");
    }
    
    // Test 7: Sum reduction
    {
        std::cout << "Test 7: Sum reduction\n";
        Tensor a_cpp({1.0, 2.0, 3.0, 4.0}, {2, 2}, true);

        Tensor c_cpp = a_cpp.sum().sum();
        c_cpp.backward();

        auto a_torch = torch::tensor({{1.0, 2.0}, {3.0, 4.0}}, torch::requires_grad());
        auto c_torch = a_torch.sum().sum();
        c_torch.backward();

        compare_tensors(c_cpp.data, c_torch, "Sum Reduction Result");
        compare_tensors(a_cpp.grad->data, a_torch.grad(), "Sum Reduction Gradient");
    }

    // Test 8: Mean reduction
    {
        std::cout << "Test 8: Mean reduction\n";
        Tensor a_cpp({1.0, 2.0, 3.0, 4.0}, {2, 2}, true);

        Tensor c_cpp = a_cpp.mean().mean();
        c_cpp.backward();

        auto a_torch = torch::tensor({{1.0, 2.0}, {3.0, 4.0}}, torch::requires_grad());
        auto c_torch = a_torch.mean().mean();
        c_torch.backward();

        compare_tensors(c_cpp.data, c_torch, "Mean Reduction Result");
        compare_tensors(a_cpp.grad->data, a_torch.grad(), "Mean Reduction Gradient");
    }

    // Test 9: Exponential
    {
        std::cout << "Test 9: Exponential\n";
        Tensor a_cpp({1.0, 2.0, 3.0, 4.0}, {2, 2}, true);

        Tensor c_cpp = a_cpp.exp().mean().mean();
        c_cpp.backward();

        auto a_torch = torch::tensor({{1.0, 2.0}, {3.0, 4.0}}, torch::requires_grad());
        auto c_torch = a_torch.exp().mean();
        c_torch.backward();

        compare_tensors(c_cpp.data, c_torch, "Exponential Result");
        compare_tensors(a_cpp.grad->data, a_torch.grad(), "Exponential Gradient");
    }

}

// Function to test batched operations
void test_batched_operations() {
    {
        std::cout << "Test 1: Batched Matrix Multiplication\n";

        // Define batch matrices for cppgrad
        Tensor a_cpp({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, {2, 2, 2}, true); // Shape: (2, 2, 2)
        Tensor b_cpp({13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0}, {2, 2, 2}, true); // Shape: (2, 2, 2)

        Tensor c_cpp = a_cpp.matmul(b_cpp); // Shape: (2, 2, 2)
        Tensor c_cpp_sum = c_cpp.sum().sum().sum(); // Reduce to scalar for backward
        c_cpp_sum.backward();

        // Define batch matrices for PyTorch
        auto a_torch = torch::tensor({{{1.0, 2.0}, {3.0, 4.0}},
                                    {{5.0, 6.0}, {7.0, 8.0}}}, torch::requires_grad()); // Shape: (2, 2, 3)
        auto b_torch = torch::tensor({{{13.0, 14.0}, {15.0, 16.0}},
                                    {{17.0, 18.0}, {19.0, 20.0}}}, torch::requires_grad()); // Shape: (2, 3, 2)

        auto c_torch = torch::bmm(a_torch, b_torch); // Shape: (2, 2, 2)
        auto c_torch_sum = c_torch.sum().sum().sum(); // Reduce to scalar for backward
        c_torch_sum.backward();

        // Compare results
        compare_tensors(c_cpp.data, c_torch, "Batched Matrix Multiplication Result");
        compare_tensors(a_cpp.grad->data, a_torch.grad(), "Batched Matrix Multiplication Gradient (a)");
        compare_tensors(b_cpp.grad->data, b_torch.grad(), "Batched Matrix Multiplication Gradient (b)");
    }    

    {
        std::cout << "Test 2: Broadcasting Addition\n";
        // Shape (2, 2, 1)
        Tensor a_cpp({1.0, 2.0, 3.0, 4.0}, {2, 2, 1}, true);
        // Shape (2)
        Tensor b_cpp({5.0, 6.0}, {2}, true);

        Tensor c_cpp = (a_cpp + b_cpp).sum().sum().sum();
        c_cpp.backward();

        auto a_torch = torch::tensor({{{1.0}, {2.0}}, {{3.0}, {4.0}}}, torch::requires_grad());
        auto b_torch = torch::tensor({5.0, 6.0}, torch::requires_grad());

        auto c_torch = (a_torch + b_torch).sum().sum();
        c_torch.backward();
        
        compare_tensors(c_cpp.data, c_torch, "Broadcasting Addition Result");
        compare_tensors(a_cpp.grad->data, a_torch.grad(), "Broadcasting Addition Gradient (a)");
        compare_tensors(b_cpp.grad->data, b_torch.grad(), "Broadcasting Addition Gradient (b)");

    }

    {
        std::cout << "Test 3: Broadcasting Multiplication\n";
        // Shape (2, 2, 1)
        Tensor a_cpp({1.0, 2.0, 3.0, 4.0}, {2, 2, 1}, true);
        // Shape (2)
        Tensor b_cpp({5.0, 6.0}, {2}, true);
        Tensor c_cpp = (a_cpp * b_cpp).sum().sum().sum();
        c_cpp.backward();

        auto a_torch = torch::tensor({{{1.0}, {2.0}}, {{3.0}, {4.0}}}, torch::requires_grad());
        auto b_torch = torch::tensor({5.0, 6.0}, torch::requires_grad());
        auto c_torch = (a_torch * b_torch).sum().sum();
        c_torch.backward(); 

        compare_tensors(c_cpp.data, c_torch, "Broadcassted Multiplication Result");
        compare_tensors(a_cpp.grad->data, a_torch.grad(), "Broadcasted Multiplication Gradient (a)");
        compare_tensors(b_cpp.grad->data, b_torch.grad(), "Broadcasted Multiplication Gradient (b)");

    }

    {
        std::cout << "Test 3: Broadcasting Division\n";
        // Shape (2, 2, 1)
        Tensor a_cpp({1.0, 2.0, 3.0, 4.0}, {2, 2, 1}, true);
        // Shape (2)
        Tensor b_cpp({5.0, 6.0}, {2}, true);
        Tensor c_cpp = (a_cpp / b_cpp).sum().sum().sum();
        c_cpp.backward();

        auto a_torch = torch::tensor({{{1.0}, {2.0}}, {{3.0}, {4.0}}}, torch::requires_grad());
        auto b_torch = torch::tensor({5.0, 6.0}, torch::requires_grad());
        auto c_torch = (a_torch / b_torch).sum().sum();
        c_torch.backward(); 

        compare_tensors(c_cpp.data, c_torch, "Broadcassted Division Result");
        compare_tensors(a_cpp.grad->data, a_torch.grad(), "Broadcasted Division Gradient (a)");
        compare_tensors(b_cpp.grad->data, b_torch.grad(), "Broadcasted Division Gradient (b)");

    }
}

void test_large_matrix_multiplication() {
    std::cout << "Test: Large Matrix Multiplication\n";

    // Generate random tensors using your custom Tensor class
    Tensor a_cpp = Tensor::randn({128, 64}, true);  // requires_grad=true
    Tensor b_cpp = Tensor::randn({64, 32}, true);
    Tensor c_cpp = Tensor::randn({32, 16}, true);

    // Convert them to PyTorch tensors while preserving requires_grad
    auto a_torch = torch::tensor(a_cpp.data, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true)).view({128, 64});
    auto b_torch = torch::tensor(b_cpp.data, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true)).view({64, 32});
    auto c_torch = torch::tensor(c_cpp.data, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true)).view({32, 16});

    // Perform matrix multiplication
    Tensor d_cpp = ((a_cpp.matmul(b_cpp)).matmul(c_cpp));;
    // d_cpp.backward();

    // Perform matrix multiplication in PyTorch
    auto d_torch = torch::matmul(torch::matmul(a_torch, b_torch), c_torch);
    // d_torch.backward();

    // Compare results
    compare_tensors(d_cpp.data, d_torch, "Matrix Multiplication Result");
    // compare_tensors(a_cpp.grad->data, a_torch.grad(), "Matrix Multiplication Gradient (a)");
    // compare_tensors(b_cpp.grad->data, b_torch.grad(), "Matrix Multiplication Gradient (b)");
    // compare_tensors(c_cpp.grad->data, c_torch.grad(), "Matrix Multiplication Gradient (c)");
}

int main() {
    std::cout << "Testing scalar operations...\n";
    test_scalars();

    std::cout << "\nTesting multidimensional operations...\n";
    test_multidimensional();

    std::cout << "\nTesting batched operations...\n";
    test_batched_operations();

    std::cout << "\nTesting large matrix multiplication...\n";
    test_large_matrix_multiplication();

    return 0;
}
