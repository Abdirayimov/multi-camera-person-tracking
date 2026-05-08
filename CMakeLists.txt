cmake_minimum_required(VERSION 3.22)
project(mc_tracking VERSION 0.1.0 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(CompilerSettings)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(BUILD_TOOLS "Build CLI tools" ON)
option(BUILD_DEEPSTREAM "Build the DeepStream pipeline (requires NvDCF + libgstreamer)" ON)
option(BUILD_TESTS "Build unit tests" OFF)

# ---- Dependencies ----
find_package(CUDAToolkit 12.0 REQUIRED)
find_package(OpenCV 4.5 REQUIRED COMPONENTS core imgproc imgcodecs videoio)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(TensorRT REQUIRED)

if(BUILD_DEEPSTREAM)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GST REQUIRED IMPORTED_TARGET
        gstreamer-1.0
        gstreamer-app-1.0
        gobject-2.0
        glib-2.0
    )
    find_package(DeepStream 8.0)
    if(NOT DeepStream_FOUND)
        message(WARNING "DeepStream SDK not found; disabling DeepStream pipeline")
        set(BUILD_DEEPSTREAM OFF)
    endif()
endif()

# ---- Library: mc_tracking (static) ----
file(GLOB_RECURSE MCT_SOURCES
    "${CMAKE_SOURCE_DIR}/src/config/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/utils/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/tracker/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/reid/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/crosscam/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/trt/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/pipeline/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/overlay/*.cpp"
)

if(NOT BUILD_DEEPSTREAM)
    list(FILTER MCT_SOURCES EXCLUDE REGEX "deepstream_pipeline\\.cpp$")
    list(FILTER MCT_SOURCES EXCLUDE REGEX "nvdcf_tracker\\.cpp$")
endif()

add_library(mc_tracking STATIC ${MCT_SOURCES})
target_include_directories(mc_tracking
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        ${OpenCV_INCLUDE_DIRS}
        ${TensorRT_INCLUDE_DIRS}
)
target_link_libraries(mc_tracking
    PUBLIC
        CUDA::cudart
        CUDA::cuda_driver
        ${OpenCV_LIBS}
        Eigen3::Eigen
        spdlog::spdlog
        yaml-cpp::yaml-cpp
        TensorRT::nvinfer
        TensorRT::nvonnxparser
)
if(BUILD_DEEPSTREAM)
    target_include_directories(mc_tracking PUBLIC ${DeepStream_INCLUDE_DIRS})
    target_link_libraries(mc_tracking PUBLIC
        DeepStream::nvds_meta
        DeepStream::nvbufsurface
        PkgConfig::GST
    )
    target_compile_definitions(mc_tracking PUBLIC MCT_HAVE_DEEPSTREAM=1)
endif()
set_target_properties(mc_tracking PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---- Executable: mc_tracking_video (offline OpenCV driver) ----
add_executable(mc_tracking_video src/main.cpp)
target_link_libraries(mc_tracking_video PRIVATE mc_tracking)

if(BUILD_TOOLS)
    add_executable(mc_tracking_benchmark tools/benchmark.cpp)
    target_link_libraries(mc_tracking_benchmark PRIVATE mc_tracking)
endif()

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

install(TARGETS mc_tracking_video RUNTIME DESTINATION bin)
if(BUILD_TOOLS)
    install(TARGETS mc_tracking_benchmark RUNTIME DESTINATION bin)
endif()
install(DIRECTORY configs/ DESTINATION share/mc_tracking/configs)

message(STATUS "")
message(STATUS "mc_tracking ${PROJECT_VERSION} configuration:")
message(STATUS "  Build type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard:     ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA:             ${CUDAToolkit_VERSION}")
message(STATUS "  OpenCV:           ${OpenCV_VERSION}")
message(STATUS "  TensorRT:         ${TensorRT_VERSION}")
message(STATUS "  DeepStream path:  ${BUILD_DEEPSTREAM}")
message(STATUS "  Build tools:      ${BUILD_TOOLS}")
message(STATUS "  Build tests:      ${BUILD_TESTS}")
message(STATUS "")
