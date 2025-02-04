#include <iostream>
#include <chrono>
#include <functional>
#include <iomanip>

#ifndef TIMER_H
#define TIMER_H

/**
 * @brief Macro to measure the execution time of a code block.
 *
 * This macro records the start and end time of a given code block
 * and prints the duration in milliseconds.
 *
 * @param code_block The block of code whose execution time will be measured.
 * @param label A string label that will be printed with the measured time.
 *
 * @note Usage:
 * @code
 * TIME_IT({
 *     // Code to measure
 * }, "My Code Block");
 * @endcode
 */
#define TIME_IT(code_block, label) {\
    auto start = std::chrono::high_resolution_clock::now(); \
    code_block \
    auto end = std::chrono::high_resolution_clock::now(); \
    std::cout << label << " took: " \
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() \
              << " ms" << std::endl; \
}

/**
 * @brief A utility class for measuring execution time of functions.
 *
 * This class provides a static `measure()` function that executes a given function
 * and measures its runtime in milliseconds. The results are printed in a format
 * similar to GoogleTest logs.
 */
class ExecutionTimer {
public:
    /**
     * @brief Measures the execution time of a function.
     *
     * Executes the provided function and measures the elapsed time in milliseconds.
     * The result is printed in a format similar to GoogleTest logs.
     *
     * @param testName The name of the test or function being measured.
     * @param func A `std::function<void()>` representing the function to execute.
     * @return float The measured execution time in milliseconds.
     *
     * @note Example usage:
     * @code
     * ExecutionTimer::measure("Sorting Algorithm", []() {
     *     std::sort(vec.begin(), vec.end());
     * });
     * @endcode
     */
    static float measure(const std::string& testName, const std::function<void()>& func) {
        auto start = std::chrono::high_resolution_clock::now();

        func();

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

        /* Print in GoogleTest-like format */
        std::cout << "[ RUN      ] " << testName << std::endl;
        std::cout << "[       OK ] " << testName << " (" << duration.count() << " ms)" << std::endl;
        return duration.count();
    }
};

#endif // TIMER_H