#include "cppgrad.h"
#include <iostream>
// supports +, -, unary -, elementwise *, and matmul
int main() {
    Tensor a({1.0, 2.0}, {3, 1, 2}, true);
    Tensor b({4.0, 1.0}, {3, 2, 1}, true);


    std::cout<<"a shape: "<<a.shape[0]<<std::endl;
    std::cout<<"a shape: "<<a.shape[1]<<std::endl;

    std::cout<<"b shape: "<<b.shape[0]<<std::endl;
    std::cout<<"b shape: "<<b.shape[1]<<std::endl;

    Tensor d = a.matmul(b);

    d.backward();

    std::cout << "Result: ";
    for (float g : d.data) {
        std::cout << g << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Gradient of a: ";
    for (float g : a.grad->data) {
        std::cout << g << " ";
    }
    std::cout << std::endl;

    std::cout << "Gradient of b: ";
    for (float g : b.grad->data) {
        std::cout << g << " ";
    }
    std::cout << std::endl;

    return 0;
}
