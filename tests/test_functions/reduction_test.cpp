#include "../unit_tests.h"

void reduction_test(const std::string& test_name, 
                     const std::function<Tensor(Tensor)>& cpp_op,
                     const std::function<torch::Tensor(torch::Tensor)>& torch_op) {
    
    Tensor a_cpp = Tensor::randn({10}, true);
    Tensor c_cpp = cpp_op(a_cpp);
    c_cpp.backward();

    auto a_torch = torch::tensor(a_cpp.data(), torch::TensorOptions().dtype(torch::kFloat32)).requires_grad_(true);
    auto c_torch = torch_op(a_torch);
    c_torch.backward();
    
    /* compare_tensors tensors needed here because we need to compare gradients as well */
    ASSERT_TRUE(compare_tensors(c_cpp.data(), c_torch, test_name + " Result"));
    ASSERT_TRUE(compare_tensors(a_cpp.grad().data(), a_torch.grad(), test_name + " Gradient (a)"));
}

void reduction_test(const std::string& test_name, 
                     const std::vector<size_t>& shape,
                     const std::function<Tensor(Tensor)>& cpp_op,
                     const std::function<torch::Tensor(torch::Tensor)>& torch_op) {
    
    Tensor a_cpp = Tensor::randn(shape, true);
    Tensor c_cpp = cpp_op(a_cpp);
    c_cpp.backward();

    std::vector<int64_t> shape_int(shape.begin(), shape.end());

    auto a_torch = torch::from_blob(a_cpp.data().data(), shape_int, torch::TensorOptions().dtype(torch::kFloat32)).clone().requires_grad_(true);
    auto c_torch = torch_op(a_torch);
    c_torch.backward();
    
    /* compare_tensors tensors needed here because we need to compare gradients as well */
    ASSERT_TRUE(compare_tensors(c_cpp.data(), c_torch, test_name + " Result"));
    ASSERT_TRUE(compare_tensors(a_cpp.grad().data(), a_torch.grad(), test_name + " Gradient (a)"));
}