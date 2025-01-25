cmake_minimum_required(VERSION 3.10)

# Project name
project(CPPGradEngine)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set CUDA Location
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

# Find LibTorch
find_package(Torch REQUIRED)

# Add the library
add_library(cppgrad STATIC cppgrad.cpp)

# Enable warnings for cppgrad
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cppgrad PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Link Torch libraries to cppgrad
target_link_libraries(cppgrad PRIVATE "${TORCH_LIBRARIES}")
target_include_directories(cppgrad PRIVATE ${TORCH_INCLUDE_DIRS})

# Add the test executable for backward.cpp
add_executable(backward_test ../../tests/backward.cpp)
target_link_libraries(backward_test PRIVATE cppgrad "${TORCH_LIBRARIES}")
target_include_directories(backward_test PRIVATE ${TORCH_INCLUDE_DIRS})

# Add the test executable for digit classification
add_executable(digit_classification ../../tests/digit_classification.cpp)
target_link_libraries(digit_classification PRIVATE cppgrad "${TORCH_LIBRARIES}")
target_include_directories(digit_classification PRIVATE ${TORCH_INCLUDE_DIRS})

# Set flags for LibTorch (optional)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Print helpful info
message(STATUS "Torch Include Path: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Torch Libraries: ${TORCH_LIBRARIES}")