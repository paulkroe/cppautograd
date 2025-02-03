#include "../unit_tests.h"

void scalar_test(const std::string& test_name, 
                     const std::function<Tensor(Tensor, Tensor)>& cpp_op, 
                     const std::function<torch::Tensor(torch::Tensor, torch::Tensor)>& torch_op) {
    
    Tensor a_cpp = Tensor::randn({1}, true);
    Tensor b_cpp = Tensor::randn({1}, true);

    Tensor c_cpp = cpp_op(a_cpp, b_cpp);
    c_cpp.backward();

    auto a_torch = torch::from_blob(a_cpp.data().data(), {1}, torch::TensorOptions().dtype(torch::kFloat32)).clone().requires_grad_(true);
    auto b_torch = torch::from_blob(b_cpp.data().data(), {1}, torch::TensorOptions().dtype(torch::kFloat32)).clone().requires_grad_(true);

    auto c_torch = torch_op(a_torch, b_torch);
    c_torch.backward();
    
    ASSERT_TRUE(compare_scalars(c_cpp.data()[0], c_torch.item<float>(), test_name + " Result"));
    ASSERT_TRUE(compare_scalars(a_cpp.grad().data()[0], a_torch.grad().item<float>(), test_name + " Gradient (a)"));
    ASSERT_TRUE(compare_scalars(b_cpp.grad().data()[0], b_torch.grad().item<float>(), test_name + " Gradient (b)"));
}