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
        if (std::abs(cpp_data[i] - torch_data[i]) > 1e-3) {
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

    {
        std::cout << "Test Exp 1: Basic Forward + Sum\n";

        // 1) Create a small tensor x with requires_grad=true
        Tensor x_cpp({0.0, 1.0, 2.0}, {3}, true);
        // 2) Apply exp
        Tensor y_cpp = x_cpp.exp();
        // 3) sum and backward
        Tensor sum_cpp = y_cpp.sum();
        sum_cpp.backward();

        auto x_torch = torch::tensor({0.0f, 1.0f, 2.0f}, torch::requires_grad());
        auto y_torch = x_torch.exp();
        auto sum_torch = y_torch.sum();
        sum_torch.backward();

        // Compare forward data
        compare_tensors(y_cpp.data, y_torch, "Exp Forward");
        // Compare gradient wrt x
        compare_tensors(x_cpp.grad->data, x_torch.grad(), "Exp Gradient (sum)");
    }

    {
        std::cout << "Test Log 1: Basic Forward + Sum\n";

        // 1) Create a small tensor x with requires_grad=true
        Tensor x_cpp({0.0, 1.0, 2.0}, {3}, true);
        // 2) Apply exp
        Tensor y_cpp = x_cpp.log();
        // 3) sum and backward
        Tensor sum_cpp = y_cpp.sum();
        sum_cpp.backward();

        auto x_torch = torch::tensor({0.0f, 1.0f, 2.0f}, torch::requires_grad());
        auto y_torch = x_torch.log();
        auto sum_torch = y_torch.sum();
        sum_torch.backward();

        // Compare forward data
        compare_tensors(y_cpp.data, y_torch, "Exp Forward");
        // Compare gradient wrt x
        compare_tensors(x_cpp.grad->data, x_torch.grad(), "Exp Gradient (sum)");
    }

    {
        std::cout << "Test Exp 2: Multiply x * exp(x) + sum\n";

        //--- C++ side ---//
        Tensor x_cpp({-1.0, 0.0, 3.0}, {3}, true);
        // e = exp(x)
        Tensor e_cpp = x_cpp.exp();
        // z = x * e
        Tensor z_cpp = x_cpp * e_cpp;  // you likely have operator* for elementwise multiply
        Tensor y_cpp = z_cpp.sum();
        y_cpp.backward();

        //--- PyTorch side ---//
        auto x_torch = torch::tensor({-1.0f, 0.0f, 3.0f}, torch::requires_grad());
        auto e_torch = x_torch.exp();
        auto z_torch = x_torch * e_torch;
        auto y_torch = z_torch.sum();
        y_torch.backward();

        // Compare forward z
        compare_tensors(z_cpp.data, z_torch, "x*exp(x) forward");
        // Compare final sum
        compare_tensors(y_cpp.data, y_torch, "x*exp(x) sum");
        // Compare gradient wrt x
        compare_tensors(x_cpp.grad->data, x_torch.grad(), "x*exp(x) gradient");
    }

    {
        std::cout << "Test Log 2: Multiply x * log(x) + sum\n";

        //--- C++ side ---//
        Tensor x_cpp({-1.0, 0.0, 3.0}, {3}, true);
        // e = log(x)
        Tensor e_cpp = x_cpp.log();
        // z = x * e
        Tensor z_cpp = x_cpp * e_cpp;  // you likely have operator* for elementwise multiply
        Tensor y_cpp = z_cpp.sum();
        y_cpp.backward();

        //--- PyTorch side ---//
        auto x_torch = torch::tensor({-1.0f, 0.0f, 3.0f}, torch::requires_grad());
        auto e_torch = x_torch.log();
        auto z_torch = x_torch * e_torch;
        auto y_torch = z_torch.sum();
        y_torch.backward();

        // Compare forward z
        compare_tensors(z_cpp.data, z_torch, "x*log(x) forward");
        // Compare final sum
        compare_tensors(y_cpp.data, y_torch, "x*log(x) sum");
        // Compare gradient wrt x
        compare_tensors(x_cpp.grad->data, x_torch.grad(), "x*log(x) gradient");
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

    // Test 10: Logarithm
    {
        std::cout << "Test 9: Logarithm\n";
        Tensor a_cpp({1.0, 2.0, 3.0, 4.0}, {2, 2}, true);

        Tensor c_cpp = a_cpp.log().mean().mean();
        c_cpp.backward();

        auto a_torch = torch::tensor({{1.0, 2.0}, {3.0, 4.0}}, torch::requires_grad());
        auto c_torch = a_torch.log().mean();
        c_torch.backward();

        compare_tensors(c_cpp.data, c_torch, "Exponential Result");
        compare_tensors(a_cpp.grad->data, a_torch.grad(), "Exponential Gradient");
    }

    {
        std::cout << "Test 10: One-hot encoding\n";
        Tensor a_cpp({0.0, 1.0, 2.0, 1.0}, {2, 2}, true);
        Tensor c_cpp = a_cpp.onehot_encode(3);

        auto a_torch = torch::tensor({{0, 1}, {2, 1}}, torch::dtype(torch::kLong));
        auto c_torch = torch::nn::functional::one_hot(a_torch, 3).to(torch::kFloat).requires_grad_(true);

        compare_tensors(c_cpp.data, c_torch, "One-hot encoding Result");
    }

    {
        std::cout << "Test 11: Cross Entropy Loss\n";
        Tensor y_pred_cpp({0.1, 0.9, 0.2, 0.8, 0.3, 0.7}, {2, 3}, true);
        Tensor y_true_cpp({0, 1}, {2}, true);

        Tensor loss_cpp = CrossEntropyLoss(y_pred_cpp, y_true_cpp);
        loss_cpp.backward();

        auto y_pred_torch = torch::tensor({{0.1, 0.9, 0.2}, {0.8, 0.3, 0.7}}, torch::requires_grad());
        auto y_true_torch = torch::tensor({0, 1}, torch::dtype(torch::kLong));
        auto loss_torch = torch::nn::functional::cross_entropy(y_pred_torch, y_true_torch).requires_grad_(true);
        loss_torch.backward();

        compare_tensors(loss_cpp.data, loss_torch, "Cross Entropy Loss Result");
        compare_tensors(y_pred_cpp.grad->data, y_pred_torch.grad(), "Cross Entropy Loss Gradient (y_pred)");
    
    }

    {
        std::cout << "Test 12: Man Cross Entropy Loss\n";

        // Ensure y_pred_cpp has requires_grad set to true
        Tensor y_pred_cpp({0.1, 0.9, 0.2, 0.8, 0.3, 0.7}, {2, 3}, true);
        Tensor y_true_cpp({0, 1}, {2}, false);

        // Apply softmax to predictions along the last dimension (classes)
        Tensor y_pred_softmax = y_pred_cpp.softmax(1);

        // Convert labels to one-hot encoding
        Tensor y_true_one_hot = y_true_cpp.onehot_encode(3);

        // Compute negative log-likelihood: - sum(one_hot * log(softmax)) over the class axis
        Tensor neg_log_likelihood = -(y_true_one_hot * y_pred_softmax.log()).sum(1);

        // Compute mean loss over batch dimension
        Tensor loss_cpp = neg_log_likelihood.mean();

        // Compute gradients
        loss_cpp.backward();


        // PyTorch Equivalent Computation
        auto y_pred_torch = torch::tensor({{0.1, 0.9, 0.2}, {0.8, 0.3, 0.7}}, torch::requires_grad());
        auto y_true_torch = torch::tensor({0, 1}, torch::dtype(torch::kLong));

        // Apply softmax along the class axis (dim=1)
        auto y_pred_softmax_torch = torch::softmax(y_pred_torch, 1);
        y_pred_softmax_torch.retain_grad();

        // Convert y_true to one-hot encoding
        auto y_true_one_hot_torch = torch::nn::functional::one_hot(y_true_torch, 3).to(torch::kFloat);

        // Compute negative log-likelihood: - sum(one_hot * log(softmax)) over class axis
        auto neg_log_likelihood_torch = -(y_true_one_hot_torch * y_pred_softmax_torch.log()).sum(1);
        neg_log_likelihood_torch.retain_grad();

        // Compute mean loss over batch dimension
        auto loss_torch = neg_log_likelihood_torch.mean();

        // Compute gradients
        loss_torch.backward();

        // Compare results
        compare_tensors(loss_cpp.data, loss_torch, "Cross Entropy Loss Result");
        compare_tensors(y_pred_cpp.grad->data, y_pred_torch.grad(), "Cross Entropy Loss Gradient (y_pred)");
        compare_tensors(y_pred_softmax.grad->data, y_pred_softmax_torch.grad(), "Cross Entropy Loss Gradient (y_pred_softmax)");
        compare_tensors(neg_log_likelihood.grad->data, neg_log_likelihood_torch.grad(), "Cross Entropy Loss Gradient (neg_log_likelihood)");

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
        std::cout << "Test 3: Broadcasting Division (1)\n";
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

    {
        std::cout << "Test 5: Broadcasting Division (3D vs 3D, partial matching)\n";
        
        // a_cpp: shape (2, 3, 4)
        Tensor a_cpp({
            // 2 batches of 3x4
            // batch 1
            1.0,  2.0,  3.0,  4.0,
            5.0,  6.0,  7.0,  8.0,
            9.0,  10.0, 11.0, 12.0,
            // batch 2
            13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0,
            21.0, 22.0, 23.0, 24.0
        }, {2, 3, 4}, true);

        // b_cpp: shape (2, 1, 4)
        Tensor b_cpp({
            // broadcast over the second dimension
            // batch 1
            2.0,  2.0,  3.0,  3.0,
            // batch 2
            4.0,  4.0,  5.0,  5.0
        }, {2, 1, 4}, true);

        // Forward
        Tensor c_cpp = (a_cpp / b_cpp).sum().sum().sum();
        c_cpp.backward();

        // Torch equivalents
        auto a_torch = torch::tensor({
            // batch 1
            { {1.0,  2.0,  3.0,  4.0},
            {5.0,  6.0,  7.0,  8.0},
            {9.0,  10.0, 11.0, 12.0} },
            // batch 2
            { {13.0, 14.0, 15.0, 16.0},
            {17.0, 18.0, 19.0, 20.0},
            {21.0, 22.0, 23.0, 24.0} }
        }, torch::requires_grad());

        auto b_torch = torch::tensor({
            // batch 1
            { {2.0, 2.0, 3.0, 3.0} },
            // batch 2
            { {4.0, 4.0, 5.0, 5.0} }
        }, torch::requires_grad());

        auto c_torch = (a_torch / b_torch).sum().sum().sum();
        c_torch.backward();

        // Compare results
        compare_tensors(c_cpp.data, c_torch, "Broadcasted Division Result (3D vs 3D)");
        compare_tensors(a_cpp.grad->data, a_torch.grad(), "Broadcasted Division Gradient (a, 3D vs 3D)");
        compare_tensors(b_cpp.grad->data, b_torch.grad(), "Broadcasted Division Gradient (b, 3D vs 3D)");
    }

    {
        std::cout << "Test 6: Broadcasting Division (tensor vs scalar)\n";

        // a_cpp: shape (3, 2)
        Tensor a_cpp({
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0
        }, {3, 2}, true);

        // b_cpp: shape (1) -> scalar 2.0
        Tensor b_cpp({2.0}, {1}, true);

        // Forward
        Tensor c_cpp = (a_cpp / b_cpp).sum().sum();
        c_cpp.backward();

        // Torch equivalents
        auto a_torch = torch::tensor({
            {1.0, 2.0},
            {3.0, 4.0},
            {5.0, 6.0}
        }, torch::requires_grad());

        auto b_torch = torch::tensor({2.0}, torch::requires_grad());
        auto c_torch = (a_torch / b_torch).sum().sum();
        c_torch.backward();

        // Compare results
        compare_tensors(c_cpp.data, c_torch, "Broadcasted Div (tensor vs scalar) - value");
        compare_tensors(a_cpp.grad->data, a_torch.grad(), "Broadcasted Div (tensor vs scalar) - grad(a)");
        compare_tensors(b_cpp.grad->data, b_torch.grad(), "Broadcasted Div (tensor vs scalar) - grad(b)");
    }

    {
        std::cout << "Test 7: Broadcasting Division (multi-dimensional)\n";

        // a_cpp: shape (1, 3, 1)
        // We'll just store 3 values: 
        // data layout: { 1.0, 2.0, 3.0 }
        Tensor a_cpp({1.0, 2.0, 3.0}, {1, 3, 1}, true);

        // b_cpp: shape (2, 1, 4)
        // We'll store 8 values in row-major:
        // Let's do something like:
        // b[0,0,:] = [2, 4, 6, 8]
        // b[1,0,:] = [1, 3, 5, 7]
        Tensor b_cpp({2.0, 4.0, 6.0, 8.0, 
                    1.0, 3.0, 5.0, 7.0}, {2, 1, 4}, true);

        // Forward operation
        // The result is shape (2, 3, 4), then we sum all elements
        Tensor c_cpp = (a_cpp / b_cpp).sum().sum().sum();
        c_cpp.backward();

        // Torch equivalents
        auto a_torch = torch::tensor({
            { {1.0}, {2.0}, {3.0} }
        }, torch::requires_grad());  // shape (1,3,1)

        // shape (2,1,4)
        auto b_torch = torch::tensor({
            { {2.0, 4.0, 6.0, 8.0} },
            { {1.0, 3.0, 5.0, 7.0} },
        }, torch::requires_grad());

        auto c_torch = (a_torch / b_torch).sum().sum().sum();
        c_torch.backward();

        // Compare
        compare_tensors(c_cpp.data, c_torch, "Broadcasted Div (1,3,1) vs (2,1,4) - value");
        compare_tensors(a_cpp.grad->data, a_torch.grad(), "Broadcasted Div (1,3,1) vs (2,1,4) - grad(a)");
        compare_tensors(b_cpp.grad->data, b_torch.grad(), "Broadcasted Div (1,3,1) vs (2,1,4) - grad(b)");
    }

    {
        std::cout << "Test 8: Broadcasting Division with extra ops\n";

        // a_cpp: shape (2, 2)
        Tensor a_cpp({1.0, 2.0, 3.0, 4.0}, {2, 2}, true);

        // b_cpp: shape (2, 1)
        Tensor b_cpp({2.0, 3.0}, {2, 1}, true);

        // Forward: do (a_cpp / b_cpp) + a_cpp, then sum
        // This checks that we can handle broadcasting in the division, 
        // but still accumulate gradient from the addition as well.
        Tensor c_cpp = ((a_cpp / b_cpp) + a_cpp).sum().sum();
        c_cpp.backward();

        // Torch equivalents
        auto a_torch = torch::tensor({{1.0, 2.0}, {3.0, 4.0}}, torch::requires_grad());
        auto b_torch = torch::tensor({{2.0}, {3.0}}, torch::requires_grad());

        auto c_torch = ((a_torch / b_torch) + a_torch).sum().sum();
        c_torch.backward();

        compare_tensors(c_cpp.data, c_torch, "Broadcast Div + Add - value");
        compare_tensors(a_cpp.grad->data, a_torch.grad(), "Broadcast Div + Add - grad(a)");
        compare_tensors(b_cpp.grad->data, b_torch.grad(), "Broadcast Div + Add - grad(b)");
    }

    {
        std::cout << "Test 4: Broadcasting Division (2)\n";
        // Shape (2, 2)
        Tensor a_cpp({1.0, 2.0, 3.0, 4.0}, {2, 2}, true);
        // Shape (2)
        Tensor b_cpp({6.0}, {1}, true);
        Tensor c_cpp = (a_cpp / b_cpp).sum().sum();
        c_cpp.backward();

        auto a_torch = torch::tensor({{1.0, 2.0}, {3.0, 4.0}}, torch::requires_grad());
        auto b_torch = torch::tensor({6.0}, torch::requires_grad());
        auto c_torch = (a_torch / b_torch).sum().sum();
        c_torch.backward(); 

        compare_tensors(c_cpp.data, c_torch, "Broadcassted Division Result (2)");
        compare_tensors(a_cpp.grad->data, a_torch.grad(), "Broadcasted Division Gradient (a) (2)");
        compare_tensors(b_cpp.grad->data, b_torch.grad(), "Broadcasted Division Gradient (b) (2)");

    }

    {
        std::cout << "Test 4: Softmax\n";
        // Shape (2, 2, 1)
        Tensor a_cpp({1.0, 2.0, 3.0, 4.0}, {4}, true);
        Tensor c_cpp = a_cpp.softmax(0).sum();
        c_cpp.backward();

        auto a_torch = torch::tensor({1.0, 2.0, 3.0, 4.0}, torch::requires_grad());
        auto c_torch = torch::nn::functional::softmax(a_torch, 0).sum();
        c_torch.backward();

        compare_tensors(c_cpp.data, c_torch, "Softmax Result");
        compare_tensors(a_cpp.grad->data, a_torch.grad(), "Softmax Gradient (a)");
    }

    {
        std::cout << "Test 1: Softmax with Mean Reduction\n";
        Tensor a_cpp({1.0, 2.0, 3.0, 4.0}, {4}, true);
        Tensor c_cpp = a_cpp.softmax(0).mean();
        c_cpp.backward();

        auto a_torch = torch::tensor({1.0, 2.0, 3.0, 4.0}, torch::requires_grad());
        auto c_torch = torch::nn::functional::softmax(a_torch, 0).mean();
        c_torch.backward();

        compare_tensors(c_cpp.data, c_torch, "Softmax Result");
        compare_tensors(a_cpp.grad->data, a_torch.grad(), "Softmax Gradient (a)");
    }

    {
        std::cout << "Test 2: Softmax Along Different Dimensions\n";
        Tensor a_cpp({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, {2, 3}, true);
        Tensor c_cpp = a_cpp.softmax(1).sum().sum();
        c_cpp.backward();

        auto a_torch = torch::tensor({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}, torch::requires_grad());
        auto c_torch = torch::nn::functional::softmax(a_torch, 1).sum().sum();
        c_torch.backward();

        compare_tensors(c_cpp.data, c_torch, "Softmax Result (dim=1)");
        compare_tensors(a_cpp.grad->data, a_torch.grad(), "Softmax Gradient (dim=1)");
    }

    {
        std::cout << "Test: Mixed Operations with Softmax and Reduction\n";

        // Create input tensor with requires_grad=true
        Tensor a_cpp({1.2, 2.3, -1.4, 3.1, 0.5, -0.8}, {2, 3}, true);

        // Perform various arithmetic operations before softmax
        Tensor b_cpp = (a_cpp * Tensor({2.0}, {1}, false) - Tensor({1.5}, {1}, false));
        Tensor c_cpp = b_cpp / Tensor({1.2}, {1}, false);
        Tensor d_cpp = c_cpp + Tensor({0.8}, {1}, false);

        // Apply softmax along the last dimension
        Tensor e_cpp = d_cpp.softmax(1);

        // Reduce by summing over class dimension
        Tensor f_cpp = e_cpp.sum(1);

        // Further operations after reduction
        Tensor g_cpp = f_cpp * Tensor({2}, {1}, false);
        Tensor h_cpp = g_cpp - Tensor({0.5}, {1}, false);
        Tensor i_cpp = h_cpp.mean(); // Final loss

        // Compute gradients
        i_cpp.backward();

        // PyTorch Equivalent Computation
        auto a_torch = torch::tensor({{1.2, 2.3, -1.4}, {3.1, 0.5, -0.8}}, torch::requires_grad());

        // Perform the same operations in PyTorch
        auto b_torch = (a_torch * 2.0) - 1.5;
        auto c_torch = b_torch / 1.2;
        auto d_torch = c_torch + 0.8;
        auto e_torch = torch::softmax(d_torch, 1);
        auto f_torch = e_torch.sum(1);
        auto g_torch = f_torch * 2.0;
        auto h_torch = g_torch - 0.5;
        auto i_torch = h_torch.mean(); // Final loss

        // Compute gradients
        i_torch.backward();

        // Compare results
        compare_tensors(i_cpp.data, i_torch, "Final Loss");
        compare_tensors(a_cpp.grad->data, a_torch.grad(), "Gradient of Input Tensor (a)");
    }

    {
        std::cout << "Test 5: Man Softmax\n";
        
        Tensor a_cpp({1.0, 2.0, 3.0, 4.0}, {4}, true);
        Tensor a_cpp_exp = a_cpp.exp();
        Tensor a_cpp_sum = a_cpp_exp.sum();
        Tensor c_cpp = (a_cpp_exp / a_cpp_sum).sum();
        c_cpp.backward();

        auto a_torch = torch::tensor({1.0, 2.0, 3.0, 4.0}, torch::requires_grad());
        auto a_torch_exp = a_torch.exp();
        a_torch_exp.retain_grad();
        auto a_torch_sum = a_torch_exp.sum();
        a_torch_sum.retain_grad();
        auto c_torch = (a_torch_exp / a_torch_sum).sum();
        c_torch.backward();
        
        compare_tensors(c_cpp.data, c_torch, "Man Softmax Result");
        compare_tensors(a_cpp_exp.grad->data, a_torch_exp.grad(), "Exp Softmax Gradient");
        // compare_tensors(a_cpp_sum.grad->data, a_torch_sum.grad(), "Sum Softmax Gradient");
        compare_tensors(a_cpp.grad->data, a_torch.grad(), "Man Softmax Gradient (a)");
    }

    // Test relu activation
    {
        Tensor a_cpp({-1.0, 2.0, -3.0, 4.0}, {4}, true);
        Tensor a_cpp_exp = a_cpp.relu().exp();
        Tensor a_cpp_sum = a_cpp_exp.sum();
        Tensor c_cpp = (a_cpp_exp / a_cpp_sum).sum();
        c_cpp.backward();

        auto a_torch = torch::tensor({-1.0, 2.0, -3.0, 4.0}, torch::requires_grad());
        auto a_torch_exp = torch::nn::functional::relu(a_torch.exp());
        a_torch_exp.retain_grad();
        auto a_torch_sum = a_torch_exp.sum();
        a_torch_sum.retain_grad();
        auto c_torch = (a_torch_exp / a_torch_sum).sum();
        c_torch.backward();
        
        compare_tensors(c_cpp.data, c_torch, "Relu Result");
        compare_tensors(a_cpp.grad->data, a_torch.grad(), "Relu Gradient (a)");
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