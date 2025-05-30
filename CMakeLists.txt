cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)
project(hello_pybind)
add_subdirectory(pybind11)
add_subdirectory(matplotplusplus)


include_directories("C:/Users/david/AppData/Local/Programs/Python/Python313/Lib/site-packages/pybind11/include")
include_directories("C:/Users/david/AppData/Local/Programs/Python/Python313")




pybind11_add_module(example example.cpp)

target_link_libraries(example PRIVATE matplot)