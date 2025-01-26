#include <torch/torch.h>
#include "../grad/cppgrad.h"
#include <iostream>
#include <cmath>
#include <gtest/gtest.h>

/* Helper functions to compare tensors */
bool compare_scalars(float cpp_value, float torch_value, const std::string& test_name);
bool compare_tensors(const std::vector<float>& cpp_data, const torch::Tensor& torch_tensor, const std::string& test_name);

/* Helper functions to run tests */
void scalar_test(const std::string& test_name, 
                     const std::function<Tensor(Tensor, Tensor)>& cpp_op, 
                     const std::function<torch::Tensor(torch::Tensor, torch::Tensor)>& torch_op);

void reduction_test(const std::string& test_name, 
                     const std::function<Tensor(Tensor)>& cpp_op,
                     const std::function<torch::Tensor(torch::Tensor)>& torch_op);

void reduction_test(const std::string& test_name, 
                     const std::vector<size_t>& shape,
                     const std::function<Tensor(Tensor)>& cpp_op,
                     const std::function<torch::Tensor(torch::Tensor)>& torch_op);

void multidim_test(const std::string& test_name,
                     const std::vector<size_t>& shape1,
                     const std::vector<size_t>& shape2,
                     const std::function<Tensor(Tensor, Tensor)>& cpp_op, 
                     const std::function<torch::Tensor(torch::Tensor, torch::Tensor)>& torch_op);

void multidim_test(const std::string& test_name,
                     const std::vector<size_t>& shape1,
                     const std::vector<size_t>& shape2,
                     const std::vector<size_t>& shape3,
                     const std::function<Tensor(Tensor, Tensor, Tensor)>& cpp_op, 
                     const std::function<torch::Tensor(torch::Tensor, torch::Tensor, torch::Tensor)>& torch_op);