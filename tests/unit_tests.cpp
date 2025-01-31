#include <gtest/gtest.h>
#include <torch/torch.h>
#include "unit_tests.h"

/* Helper function to compare scalar tensors */
bool compare_scalars(float cpp_value, float torch_value, const std::string& test_name) {
    if (std::abs(cpp_value - torch_value) < 1e-6) {
        return true;
    } else {
        std::cerr << test_name << " FAILED: " << cpp_value << " (cpp) vs " << torch_value << " (torch).\n";
        return false;
    }
}

/* Helper function to compare tensors */
bool compare_tensors(const std::vector<float>& cpp_data, const torch::Tensor& torch_tensor, const std::string& test_name) {
    auto torch_data = torch_tensor.flatten().data_ptr<float>();
    for (size_t i = 0; i < cpp_data.size(); ++i) {
        if (std::abs(cpp_data[i] - torch_data[i]) > 1e-3) {
            std::cerr << test_name << " FAILED at index " << i << ": " << cpp_data[i] << " (cpp) vs " << torch_data[i] << " (torch).\n";
            std::cerr << "Difference: " << std::abs(cpp_data[i] - torch_data[i]) << std::endl;
            return false;
        }
    }
    return true;
}


/* BEGIN SCALAR TESTS */
TEST(TensorTest, ScalarAddition) {
    scalar_test("Scalar Addition", 
        [](Tensor a, Tensor b) { return a + b; },
        [](torch::Tensor a, torch::Tensor b) { return a + b; });
}

TEST(TensorTest, ScalarMultiplication) {
    scalar_test("Scalar Multiplication", 
        [](Tensor a, Tensor b) { return a * b; },
        [](torch::Tensor a, torch::Tensor b) { return a * b; });
}

TEST(TensorTest, ScalarDivision) {
    scalar_test("Scalar Division", 
        [](Tensor a, Tensor b) { return a / b; },
        [](torch::Tensor a, torch::Tensor b) { return a / b; });
}
/* END SCALAR TESTS */



/* BEGIN REDUCTION TESTS */
TEST(TensorTest, Mean) {
    reduction_test("Mean", 
        [](Tensor a) { return a.mean(); },
        [](torch::Tensor a) { return a.mean(); });
}

TEST(TensorTest, DoubleMean) {
    reduction_test("Double Mean", 
        {10, 10},
        [](Tensor a) { return a.exp().mean().mean(); },
        [](torch::Tensor a) { return a.exp().mean(); });
}

TEST(TensorTest, Sum) {
    reduction_test("Sum", 
        [](Tensor a) { return a.sum(); },
        [](torch::Tensor a) { return a.sum(); });
}

TEST(TensorTest, DoubleSum) {
    reduction_test("Double Sum",
        {10, 10},
        [](Tensor a) { return a.log().sum().sum(); },
        [](torch::Tensor a) { return a.log().sum(); });
}
/* END REDUCTION TESTS */

/* BEGIN OPERATOR TESTS */
TEST(TensorTest, Exponential) {
    reduction_test("Exponential", 
        [](Tensor a) { return a.exp().mean(); },
        [](torch::Tensor a) { return a.exp().mean(); });
}

TEST(TensorTest, MixedExponential) {
    reduction_test("Multiply Exponential", 
        [](Tensor a) { 
            Tensor e = a.exp();
            Tensor z = a * e;
            return z.sum();
        },
        [](torch::Tensor a) { 
            auto e = a.exp();
            auto z = a * e;
            return z.sum();
        }
    );
}

TEST(TensorTest, Logarithm) {
    reduction_test("Logarithm", 
        [](Tensor a) { return a.log().mean(); },
        [](torch::Tensor a) { return a.log().mean(); });
}

TEST(TensorTest, MixedLogarithm) {
    reduction_test("Multiply Logarithm", 
        [](Tensor a) { 
            Tensor e = a.exp();
            Tensor z = a * e;
            return z.sum();
        },
        [](torch::Tensor a) { 
            auto e = a.exp();
            auto z = a * e;
            return z.sum();
        }
    );
}
/* END OPERATOR TESTS */

/* BEGIN LOSS FUNCTION TESTS */
TEST(TensorTest, Softmax) {
    reduction_test("Softmax",
        {4},
        [](Tensor a) { return a.softmax(0).sum(); },
        [](torch::Tensor a) { return torch::nn::functional::softmax(a, 0).sum(); }
    );
}

TEST(TensorTest, ReduceSoftmax) {
    reduction_test("Reducing Softmax",
        {2, 4},
        [](Tensor a) { return a.softmax(1).sum().sum(); },
        [](torch::Tensor a) { return torch::nn::functional::softmax(a, 1).sum().sum(); }
    );
}

TEST(TensorTest, MixedSoftmax) {
    reduction_test("Mixed Softmax",
        {2, 4},
        [](Tensor a) {
            Tensor b = (a * Tensor({2.0}, {1}, false) - Tensor({1.5}, {1}, false));
            Tensor c = b / Tensor({1.2}, {1}, false);
            Tensor d = c + Tensor({0.8}, {1}, false);
            Tensor e = d.softmax(1);
            Tensor f = e.sum(1);
            Tensor g = f * Tensor({2}, {1}, false);
            Tensor h = g - Tensor({0.5}, {1}, false);
            return h.mean();
        },
        [](torch::Tensor a) {
            auto b = (a * 2.0 - 1.5);
            auto c = b / 1.2;
            auto d = c + 0.8;
            auto e = torch::nn::functional::softmax(d, 1);
            auto f = e.sum(1);
            auto g = f * 2.0;
            auto h = g - 0.5;
            return h.mean();
        }
    );
}

TEST(TensorTest, ManSoftmax) {
    reduction_test("Man Softmax",
        {4},
        [](Tensor a) {
            Tensor a_exp = a.exp();
            Tensor a_sum = a_exp.sum();
            return (a_exp / a_sum).sum();
        },
        [](torch::Tensor a) {
            auto a_exp = a.exp();
            auto a_sum = a_exp.sum();
            return (a_exp / a_sum).sum();
        }
    );
}

TEST(TensorTest, ManCrossEntropyLoss) {
    reduction_test("Man Cross Entropy Loss",
        {3, 42},
        [](Tensor a) { 
            Tensor y_true({32, 10, 2}, {3}, false);
            Tensor y_pred_softmax = a.softmax(a.shape.size() - 1);
            Tensor y_true_one_hot = y_true.onehot_encode(a.shape.back());
            Tensor neg_log_likelihood = -(y_true_one_hot * y_pred_softmax.log()).sum(a.shape.size() - 1);
            return neg_log_likelihood.mean();
        },
        [](torch::Tensor a) {
            auto y_true = torch::tensor({32, 10, 2}, torch::dtype(torch::kLong));
            auto y_pred_softmax = torch::nn::functional::softmax(a, a.dim() - 1);
            auto y_true_one_hot = torch::nn::functional::one_hot(y_true, a.size(a.dim() - 1)).to(torch::kFloat);
            auto neg_log_likelihood = -(y_true_one_hot * y_pred_softmax.log()).sum(a.dim() - 1);
            return neg_log_likelihood.mean();
        }
    );
}

TEST(TensorTest, CrossEntropyLoss) {
    reduction_test("Cross Entropy Loss",
        {3, 42},
        [](Tensor a) { 
            Tensor y_true({32, 10, 2}, {3}, false);
            return CrossEntropyLoss(a, y_true);
        },
        [](torch::Tensor a) {
            auto y_true = torch::tensor({32, 10, 2}, torch::dtype(torch::kLong));
            return torch::nn::functional::cross_entropy(a, y_true);
        }
    );
}
/* END LOSS FUNCTION TESTS */

/* START ACTIVATION FUNCTION TESTS */
TEST(TensorTest, ReLU) {
    reduction_test("ReLU",
        {10},
        [](Tensor a) {
            Tensor a_exp = a.relu().exp();
            Tensor a_sum = a_exp.sum();
            return (a_exp / a_sum).sum();
        },
        [](torch::Tensor a) { 
            auto a_exp = a.relu().exp();
            auto a_sum = a_exp.sum();
            return (a_exp / a_sum).sum();
        }
    );
}
/* END ACTIVATION FUNCTION TESTS */

/* BEGIN MULTIDIMENSIONAL TESTS */
TEST(TensorTest, Addition) {
    multidim_test("Addition", 
        {10},
        {10},
        [](Tensor a, Tensor b) { return (a + b).sum(); },
        [](torch::Tensor a, torch::Tensor b) { return (a + b).sum(); });
}

TEST(TensorTest, Multiplication) {
    multidim_test("Multiplication", 
        {10},
        {10},
        [](Tensor a, Tensor b) { return (a * b).sum(); },
        [](torch::Tensor a, torch::Tensor b) { return (a * b).sum(); });
}

TEST(TensorTest, Division) {
    multidim_test("Division",
        {10},
        {10},
        [](Tensor a, Tensor b) { return (a / b).sum(); },
        [](torch::Tensor a, torch::Tensor b) { return (a / b).sum(); });
}

TEST(TensorTest, MatrixVectorMultiplication) {
    multidim_test("Matrix Vector Multiplication",
        {2, 2},
        {2, 1},
        [](Tensor a, Tensor b) { return a.matmul(b).sum().sum(); },
        [](torch::Tensor a, torch::Tensor b) { return torch::matmul(a, b).sum(); });
}

TEST(TensorTest, MatrixMultiplication) {
    multidim_test("Matrixmultiplication",
        {5, 5},
        {5, 3},
        [](Tensor a, Tensor b) { return a.matmul(b).sum().sum(); },
        [](torch::Tensor a, torch::Tensor b) { return torch::matmul(a, b).sum(); });
}

TEST(TensorTest, DoubleMatrixMultiplication) {
    multidim_test("Double Matrixmultiplication",
        {5, 5},
        {5, 3},
        {3, 10},
        [](Tensor a, Tensor b, Tensor c) { return a.matmul(b).matmul(c).sum().sum(); },
        [](torch::Tensor a, torch::Tensor b, torch::Tensor c) { return torch::matmul(torch::matmul(a, b), c).sum(); });
}
/* END MULTIDIMENSIONAL TESTS */

/* BEGIN BATCHING AND BROADCASTING TESTS */
TEST(TensorTest, BatchedMatrixMultiplication) {
    multidim_test("Batched Matrixmultiplication",
        {5, 5, 5},
        {5, 5, 5},
        [](Tensor a, Tensor b) { return a.matmul(b).sum().sum().sum(); },
        [](torch::Tensor a, torch::Tensor b) { return torch::bmm(a, b).sum(); });
}

TEST(TensorTest, BatchedMatrixAddition) {
    multidim_test("Broadcasted Batched Matrix Addition",
        {5, 5, 1},
        {5},
        [](Tensor a, Tensor b) { return (a + b).sum().sum().sum(); },
        [](torch::Tensor a, torch::Tensor b) { return (a + b).sum(); });
}

TEST(TensorTest, BatchedElementwiseMultiplication) {
    multidim_test("Broadcasted Batched Elementwise Multiplication",
        {2, 2, 1},
        {2},
        [](Tensor a, Tensor b) { return (a * b).sum().sum().sum(); },
        [](torch::Tensor a, torch::Tensor b) { return (a * b).sum(); });
}

TEST(TensorTest, BatchedElementwiseDivision) {
    multidim_test("Broadcasted Batched Elementwise Division",
        {2, 2, 1},
        {2},
        [](Tensor a, Tensor b) { return (a / b).sum().sum().sum(); },
        [](torch::Tensor a, torch::Tensor b) { return (a / b).sum().sum(); });
}

TEST(TensorTest, BatchedElementwiseDivisionTwo) {
    multidim_test("Broadcasted Batched Elementwise Division",
        {2, 3, 4},
        {2, 1, 4},
        [](Tensor a, Tensor b) { return (a / b).sum().sum().sum(); },
        [](torch::Tensor a, torch::Tensor b) { return (a / b).sum().sum(); });
}

TEST(TensorTest, TensorScalarDivision) {
    multidim_test("Tensor Scalar Division",
        {2, 3, 4},
        {1},
        [](Tensor a, Tensor b) { return (a / b).sum().sum().sum(); },
        [](torch::Tensor a, torch::Tensor b) { return (a / b).sum().sum(); });
}

TEST(TensorTest, BroadcastingElementwiseDivision) {
    multidim_test("Broadcasting Elementwise Division",
        {1, 3, 1},
        {2, 1, 4},
        [](Tensor a, Tensor b) { return (a / b).sum().sum().sum(); },
        [](torch::Tensor a, torch::Tensor b) { return (a / b).sum().sum(); });
}

TEST(TensorTest, BroadcastingElementwiseDivisionTwo) {
    multidim_test("Broadcasting Elementwise Division",
        {2, 2},
        {2},
        [](Tensor a, Tensor b) { return ((a / b) + a).sum().sum(); },
        [](torch::Tensor a, torch::Tensor b) { return ((a / b) + a).sum().sum(); });
}

TEST(TensorTest, MixedBroadcastingElementwiseDivision) {
    multidim_test("Mixed Broadcasting Elementwise Division",
        {2, 2},
        {1},
        [](Tensor a, Tensor b) { return ((a / b) + a).sum().sum(); },
        [](torch::Tensor a, torch::Tensor b) { return ((a / b) + a).sum().sum(); });
}
/* END BATCHING AND BROADCASTING TESTS */

/* main functions gtest */
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    /* Set random seed */
    set_seed(42);
// {
//     Tensor x = Tensor({0,0,0}, {3}, true);
//     Tensor exp_tensor = x.exp();
//     Tensor sum_exp = exp_tensor.sum() + 1e-6;
//     Tensor out = exp_tensor / sum_exp;
//     Tensor out_sum = out.sum();
//     out_sum.backward(); 

//     // Print gradients
//     std::cout << "Gradient x " << x.id << ": " << x.grad() << std::endl;
//     std::cout << "Gradient exp_tensor "<< exp_tensor.id << ": " << exp_tensor.grad() << std::endl;
//     std::cout << "Gradient sum_exp "<< sum_exp.id <<": " << sum_exp.grad() << std::endl;
//     std::cout << "Gradient out " << out.id << ": " << out.grad() << std::endl;

// }
// std::cout << "<---------------------------------->" << std::endl;
// {
//     auto x = torch::tensor({0.0, 0.0, 0.0}, torch::requires_grad(true));

//     // Compute softmax-related operations
//     auto exp_tensor = x.exp();
//     exp_tensor.retain_grad(); // Retain gradient for printing

//     auto sum_exp = exp_tensor.sum() + 1e-6;

//     // Reshape sum_exp for broadcasting
//     sum_exp = sum_exp.view({1}); 
//     sum_exp.retain_grad(); // Retain gradient for printing

//     auto out = exp_tensor / sum_exp;
//     out.retain_grad(); // Retain gradient for printing

//     auto out_sum = out.sum();
//     out_sum.backward(); 

//     std::cout << "Gradient x: " << x.grad() << std::endl;
//     std::cout << "Gradient exp_tensor: " << exp_tensor.grad() << std::endl;
//     std::cout << "Gradient sum_exp: " << sum_exp.grad() << std::endl;
//     std::cout << "Gradient out: " << out.grad() << std::endl;
// }

//     return 0;
    return RUN_ALL_TESTS();
}
