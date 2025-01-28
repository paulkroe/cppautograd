#include <iostream>
#include <chrono>
#include <functional>
#include <iomanip>
/* timer class used for measurements */
class ExecutionTimer {
public:
    static float measure(const std::string& testName, const std::function<void()>& func) {
        auto start = std::chrono::high_resolution_clock::now();

        func();

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

        /* print in gtest like format */
        std::cout << "[ RUN      ] " << testName << std::endl;
        std::cout << "[       OK ] " << testName << " (" << duration.count() << " ms)" << std::endl;
        return duration.count();
    }
};