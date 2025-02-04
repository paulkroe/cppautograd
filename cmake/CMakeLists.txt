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

# Set project root directory
set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

# Add options for profiling
option(ENABLE_GPROF "Enable gprof profiling" OFF)
option(ENABLE_VALGRIND "Enable Valgrind memory debugging" OFF)

# Add profiling compiler flags
if (ENABLE_GPROF)
    message(STATUS "Enabling gprof profiling")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
endif()

if (ENABLE_VALGRIND)
    message(STATUS "Enabling Valgrind debugging")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")  # Disable optimizations for better Valgrind analysis
endif()

# Add the library
add_library(cppgrad STATIC ${PROJECT_ROOT_DIR}/grad/operator_overloads.cpp
                           ${PROJECT_ROOT_DIR}/grad/backward.cpp
                           ${PROJECT_ROOT_DIR}/grad/matmul.cpp
                           ${PROJECT_ROOT_DIR}/grad/reduction_operators.cpp
                           ${PROJECT_ROOT_DIR}/grad/unary_operators.cpp
                           ${PROJECT_ROOT_DIR}/grad/loss_functions.cpp
                           ${PROJECT_ROOT_DIR}/grad/activation_functions.cpp
                           ${PROJECT_ROOT_DIR}/grad/initialization_functions.cpp
                           ${PROJECT_ROOT_DIR}/grad/util.cpp)

# Enable warnings for cppgrad
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cppgrad PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Link Torch libraries to cppgrad
target_link_libraries(cppgrad PRIVATE "${TORCH_LIBRARIES}")
target_include_directories(cppgrad PRIVATE ${TORCH_INCLUDE_DIRS} ${PROJECT_ROOT_DIR}/grad)

# Add unit tests executable
find_package(GTest REQUIRED)
add_executable(unit_tests ${PROJECT_ROOT_DIR}/tests/unit_tests.cpp
                         ${PROJECT_ROOT_DIR}/tests/test_functions/scalar_test.cpp
                         ${PROJECT_ROOT_DIR}/tests/test_functions/reduction_test.cpp
                         ${PROJECT_ROOT_DIR}/tests/test_functions/multidim_test.cpp)
target_link_libraries(unit_tests PRIVATE GTest::GTest GTest::Main cppgrad "${TORCH_LIBRARIES}")
target_include_directories(unit_tests PRIVATE ${TORCH_INCLUDE_DIRS} ${PROJECT_ROOT_DIR}/tests)

# Add the test executable for digit classification
add_executable(digit_classification ${PROJECT_ROOT_DIR}/demo/digit_classification.cpp)
target_link_libraries(digit_classification PRIVATE cppgrad "${TORCH_LIBRARIES}")
target_include_directories(digit_classification PRIVATE ${TORCH_INCLUDE_DIRS} ${PROJECT_ROOT_DIR}/demo)

# Add the tutorial executable
add_executable(tutorial ${PROJECT_ROOT_DIR}/demo/tutorial.cpp)
target_link_libraries(tutorial PRIVATE cppgrad "${TORCH_LIBRARIES}")
target_include_directories(tutorial PRIVATE ${TORCH_INCLUDE_DIRS} ${PROJECT_ROOT_DIR}/demo)

# Set flags for LibTorch
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Custom target for gprof profiling
add_custom_target(profile_gprof
    COMMAND cmake --build . --target digit_classification
    COMMAND ./digit_classification
    COMMAND gprof ./digit_classification gmon.out > gprof_report.txt
    COMMENT "Running gprof profiling..."
)

# Custom target for Valgrind memory profiling
add_custom_target(profile_valgrind
    COMMAND cmake --build . --target digit_classification
    COMMAND valgrind --leak-check=full --show-leak-kinds=all --log-file=valgrind_report.txt ./digit_classification
    COMMENT "Running Valgrind profiling..."
)
