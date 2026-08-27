cmake_minimum_required(VERSION 3.16)
project(TensorPlusPlus)

set(CMAKE_CXX_STANDARD 17)

add_executable(TensorPlusPlus main.cpp Tensor.cpp)

add_executable(TensorTests tests.cpp Tensor.cpp)
