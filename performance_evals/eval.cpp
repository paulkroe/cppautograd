#include "timer.h"
#include "../grad/cppgrad.h"

void matmul_test(size_t num_threads, size_t size) {
    Tensor a = Tensor::randn({size, size}, false);
    Tensor b = Tensor::randn({size, size}, false);
    Tensor c = a.matmul(b);
}

void matmul_backward_test(size_t num_threads, size_t size) {
    Tensor a = Tensor::randn({size, size}, false);
    Tensor b = Tensor::randn({size, size}, false);
    Tensor c = a.matmul(b).sum().sum();
    c.backward();
}

int main() {
    std::cout << "==================== Execution Timing ====================" << std::endl;
    
    float avg = 0.0f;
    std::vector<int> num_threads = {1, 2, 4, 8, 16};
    std::vector<size_t> sizes = {10, 100, 1000, 5000};
    size_t num_runs = 5;
    for (auto s: sizes) {
        for (auto t: num_threads) {
            avg = 0.0f;
            for (int i = 0; i < num_runs; i++) {
                avg += ExecutionTimer::measure("matmul_test", [s, t]() {      matmul_test(t, s);    });
            }
            std::cout << "=========================================================" << std::endl;
            std::cout << "Average time for matmul with size " << s << " and " << t << " threads: " << avg / num_runs << " ms" << std::endl;
        }

    }

    return 0;
}