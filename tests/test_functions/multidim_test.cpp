#include "../unit_tests.h"

void multidim_test(const std::string& test_name,
                     const std::vector<size_t>& shape1,
                     const std::vector<size_t>& shape2,
                     const std::function<Tensor(Tensor, Tensor)>& cpp_op, 
                     const std::function<torch::Tensor(torch::Tensor, torch::Tensor)>& torch_op) {
    
    Tensor a_cpp = Tensor::randn(shape1, true);
    Tensor b_cpp = Tensor::randn(shape2, true);

    Tensor c_cpp = cpp_op(a_cpp, b_cpp);
    c_cpp.backward();

    std::vector<int64_t> shape1_int(shape1.begin(), shape1.end());
    std::vector<int64_t> shape2_int(shape2.begin(), shape2.end());

    auto a_torch = torch::from_blob(a_cpp.data.data(), shape1_int, torch::TensorOptions().dtype(torch::kFloat32)).clone().requires_grad_(true);
    auto b_torch = torch::from_blob(b_cpp.data.data(), shape2_int, torch::TensorOptions().dtype(torch::kFloat32)).clone().requires_grad_(true);

    auto c_torch = torch_op(a_torch, b_torch);
    c_torch.backward();
    
    ASSERT_TRUE(compare_tensors(c_cpp.data, c_torch, test_name + " Result"));
    ASSERT_TRUE(compare_tensors(a_cpp.grad().data, a_torch.grad(), test_name + " Gradient (a)"));
    ASSERT_TRUE(compare_tensors(b_cpp.grad().data, b_torch.grad(), test_name + " Gradient (b)"));
}

void multidim_test(const std::string& test_name,
                     const std::vector<size_t>& shape1,
                     const std::vector<size_t>& shape2,
                     const std::vector<size_t>& shape3,
                     const std::function<Tensor(Tensor, Tensor, Tensor)>& cpp_op, 
                     const std::function<torch::Tensor(torch::Tensor, torch::Tensor, torch::Tensor)>& torch_op) {
    
    Tensor a_cpp = Tensor::randn(shape1, true);
    Tensor b_cpp = Tensor::randn(shape2, true);
    Tensor c_cpp = Tensor::randn(shape3, true);

    Tensor d_cpp = cpp_op(a_cpp, b_cpp, c_cpp);
    d_cpp.backward();

    std::vector<int64_t> shape1_int(shape1.begin(), shape1.end());
    std::vector<int64_t> shape2_int(shape2.begin(), shape2.end());
    std::vector<int64_t> shape3_int(shape3.begin(), shape3.end());

    auto a_torch = torch::from_blob(a_cpp.data.data(), shape1_int, torch::TensorOptions().dtype(torch::kFloat32)).clone().requires_grad_(true);
    auto b_torch = torch::from_blob(b_cpp.data.data(), shape2_int, torch::TensorOptions().dtype(torch::kFloat32)).clone().requires_grad_(true);
    auto c_torch = torch::from_blob(c_cpp.data.data(), shape3_int, torch::TensorOptions().dtype(torch::kFloat32)).clone().requires_grad_(true);

    auto d_torch = torch_op(a_torch, b_torch, c_torch);
    d_torch.backward();
    
    ASSERT_TRUE(compare_tensors(d_cpp.data, d_torch, test_name + " Result"));
    ASSERT_TRUE(compare_tensors(a_cpp.grad().data, a_torch.grad(), test_name + " Gradient (a)"));
    ASSERT_TRUE(compare_tensors(b_cpp.grad().data, b_torch.grad(), test_name + " Gradient (b)"));
    ASSERT_TRUE(compare_tensors(c_cpp.grad().data, c_torch.grad(), test_name + " Gradient (c)"));
}