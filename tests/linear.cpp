#include "../vanilla/cppgrad.h"
#include "../vanilla/modules.h"

/* as a first simple test train a model that always predicts 0*/
int main () {
    Linear linear(5, 1);

    for (int i = 0; i < 100; i++) {
        Tensor x = Tensor::randn({1, 5}, true);
        /* forward pass */
        Tensor y = linear.forward(x);
        Tensor loss = (y * y).sum();

        std::cout << "Loss: " << loss.data[0] << std::endl;
        
        /* backward pass */
        loss.backward();

        /* update step */
        for (size_t i = 0; i < linear.weight.data.size(); i++) {
            linear.weight.data[i] -= 0.01 * linear.weight.grad->data[i];
        }
        for (size_t i = 0; i < linear.bias.data.size(); i++) {
            linear.bias.data[i] -= 0.01 * linear.bias.grad->data[i];
        }

        /* zero the gradients */
        linear.weight.zero_grad();
        linear.bias.zero_grad();
    }
}