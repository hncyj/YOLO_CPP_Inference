cmake_minimum_required(VERSION 3.10)

project(ONNXDecoder)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加include目录
include_directories(${CMAKE_SOURCE_DIR}/include)

# 查找OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 查找OpenVINO
find_package(OpenVINO REQUIRED)
include_directories("/usr/include/openvino")

# 设置ONNX Runtime路径
set(ONNXRUNTIME_INCLUDE_DIRS "/usr/local/include/onnxruntime")
set(ONNXRUNTIME_LIB_DIRS "/usr/local/lib")

include_directories(${ONNXRUNTIME_INCLUDE_DIRS})
link_directories(${ONNXRUNTIME_LIB_DIRS})

# 收集所有源文件
set(SOURCES
    src/main.cpp
    src/ModelLoader.cpp
    src/YOLOPipeline.cpp
    src/YOLOProcessor.cpp
    src/YOLOutils.cpp
)

# 创建可执行文件
add_executable(yolo_pipeline ${SOURCES})

# 查找ONNX Runtime库
find_library(ONNXRUNTIME_LIB onnxruntime PATHS ${ONNXRUNTIME_LIB_DIRS})
if(NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "Cannot find ONNX Runtime library. Please specify the correct path.")
endif()

# 链接库
target_link_libraries(
    yolo_pipeline
    ${OpenCV_LIBS}
    openvino::runtime
    ${ONNXRUNTIME_LIB}
)

# # 可选：复制模型和资源文件到构建目录
# file(COPY ${CMAKE_SOURCE_DIR}/models DESTINATION ${CMAKE_BINARY_DIR})
# file(COPY ${CMAKE_SOURCE_DIR}/resources DESTINATION ${CMAKE_BINARY_DIR})

# 调试信息
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenVINO include dirs: ${OpenVINO_INCLUDE_DIRS}")
message(STATUS "ONNX Runtime include dirs: ${ONNXRUNTIME_INCLUDE_DIRS}")
message(STATUS "ONNX Runtime lib dirs: ${ONNXRUNTIME_LIB_DIRS}")
message(STATUS "Source files: ${SOURCES}")