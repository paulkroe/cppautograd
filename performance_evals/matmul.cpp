#include "timer.h"
#include "../grad/cppgrad.h"

void task1(size_t num_threads) {
    Tensor a = Tensor::randn({10000, 10000}, false);
    Tensor b = Tensor::randn({10000, 10000}, false);
    Tensor c = a.matmul(b, num_threads);
}

void task2() {
    Tensor a = Tensor::randn({1000, 1000}, false);
    Tensor b = Tensor::randn({1000, 1000}, false);
    Tensor c = a * b;
}

int main() {
    std::cout << "==================== Execution Timing ====================" << std::endl;
    
    int num_runs = 5;
    float avg = 0.0f;

    for (int i = 0; i < num_runs; i++) {
        avg += ExecutionTimer::measure("Task 1", [num_runs]() {  task1(num_runs);    });
        // ExecutionTimer::measure("Task 2", task2);
    }
    std::cout << "=========================================================" << std::endl;
    std::cout << "Average time for Task 1: " << avg / num_runs << " ms" << std::endl;
    return 0;
}