cmake_minimum_required(VERSION 3.5)

project(tkdnn_demo)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC -Wno-deprecated-declarations -Wno-misleading-indentation -Wno-sign-compare -Wno-unused-variable")
add_definitions("-Wall")
add_definitions(-DUNICODE -D_UNICODE)

# CUDA
find_package(CUDA 9.0 REQUIRED)
SET(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --maxrregcount=32)
include_directories(${CUDA_INCLUDE_DIRS})

# CUDNN
find_package(CUDNN REQUIRED)
include_directories(${CUDNN_INCLUDE_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Used for I/O and other image processing operations
find_package(OpenCV REQUIRED)

# TKDNN - object detection
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/tkdnn/include)
file(GLOB TKDNN_LIBS ${CMAKE_CURRENT_SOURCE_DIR}/tkdnn/lib/*.so)

set(LIBS cudart ${OpenCV_LIBS} ${TKDNN_LIBS})

# set source files
# file(GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/darknet_to_tensorrt.cpp)
file(GLOB SRC2 ${CMAKE_CURRENT_SOURCE_DIR}/src/tkdnn_wrapper.cpp)

# executable
add_executable(darknet_to_tensorrt ${SRC})
target_link_libraries(darknet_to_tensorrt ${LIBS})

add_executable(tkdnn_wrapper ${SRC2})
target_link_libraries(tkdnn_wrapper ${LIBS})
