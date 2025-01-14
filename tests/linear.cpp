#include "../vanilla/cppgrad.h"
#include "../vanilla/modules.h"

/* as a first simple test train a model that always predicts 0*/
int main () {
    Linear linear1(5, 5);
    Linear linear2(5, 1);
    
    std::cout << "weight 1: ";
    std::cout << linear1.weight << std::endl;
    std::cout << "bias 1: ";
    std::cout << linear1.bias << std::endl;
    
    std::cout << "weight 2: ";
    std::cout << linear2.weight << std::endl;
    std::cout << "bias 2: ";
    std::cout << linear2.bias << std::endl;
    float lr = 0.01;

    for (int i = 0; i < 10000; i++) {
        Tensor x = Tensor::randn({5, 1, 5}, false);
        /* forward pass */
        Tensor y = linear2.forward(linear1.forward(x));
        Tensor loss = (y * y).sum().sum().sum();

        if (i%500 ==0) {
            std::cout << "Epoch " << i << std::endl;
            std::cout << "Loss: " << loss.data[0] << std::endl;
            lr /= 2;
        }
        
        /* backward pass */
        loss.backward();

        /* update step */
        for (size_t i = 0; i < linear1.weight.data.size(); i++) {
            linear1.weight.data[i] -= lr * linear1.weight.grad->data[i];
        }
        for (size_t i = 0; i < linear1.bias.data.size(); i++) {
            linear1.bias.data[i] -= lr * linear1.bias.grad->data[i];
        }

        for (size_t i = 0; i < linear2.weight.data.size(); i++) {
            linear2.weight.data[i] -= lr * linear2.weight.grad->data[i];
        }
        for (size_t i = 0; i < linear2.bias.data.size(); i++) {
            linear2.bias.data[i] -= lr * linear2.bias.grad->data[i];
        }

        /* zero the gradients */
        linear1.weight.zero_grad();
        linear1.bias.zero_grad();
        
        linear2.weight.zero_grad();
        linear2.bias.zero_grad();
    }

    std::cout << "Final weight 1: ";
    std::cout << linear1.weight << std::endl;
    std::cout << "Final bias 1: ";
    std::cout << linear1.bias << std::endl;

    std::cout << "Final weight 2: ";
    std::cout << linear2.weight << std::endl;
    std::cout << "Final bias 2: ";
    std::cout << linear2.bias << std::endl;
}