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
option(MCT_BUILD_TESTS "Build the GoogleTest unit test suite" OFF)
option(MCT_CPU_ONLY
       "Build only the CPU-only algorithmic core (no CUDA / TensorRT)" OFF)

# Accepted for backwards compatibility with pre-0.2 build scripts.
if(BUILD_TESTS AND NOT MCT_BUILD_TESTS)
    message(WARNING "BUILD_TESTS is deprecated; use MCT_BUILD_TESTS")
    set(MCT_BUILD_TESTS ON)
endif()

if(MCT_CPU_ONLY)
    # The GPU stages have no meaning without their SDKs; turn them off
    # rather than letting find_package fail late.
    set(BUILD_TOOLS OFF)
endif()

# ---- Dependencies ----
# The algorithmic core needs only these four; the rest are GPU stages and
# are looked up separately below.
find_package(OpenCV 4.5 REQUIRED COMPONENTS core imgproc)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)

if(NOT MCT_CPU_ONLY)
    find_package(CUDAToolkit 12.0 REQUIRED)
    find_package(OpenCV 4.5 REQUIRED COMPONENTS core imgproc imgcodecs videoio)
    find_package(TensorRT REQUIRED)
endif()

# ---- Library: mc_tracking_core (static, CPU only) ----
# Config parsing, the Kalman filter, both CPU trackers, the Hungarian
# solver, the ReID gallery and the cross-camera matcher are pure CPU work.
# Keeping them in their own target lets the unit tests -- and CI runners
# without an NVIDIA stack -- build them without CUDA or TensorRT.
add_library(mc_tracking_core STATIC
    src/config/system_config.cpp
    src/utils/logger.cpp
    src/tracker/kalman_filter.cpp
    src/tracker/iou_tracker.cpp
    src/tracker/bytetrack.cpp
    src/tracker/tracker_factory.cpp
    src/crosscam/hungarian.cpp
    src/crosscam/identity_matcher.cpp
    src/reid/reid_gallery.cpp
)
target_include_directories(mc_tracking_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        ${OpenCV_INCLUDE_DIRS}
)
# spdlog compiles out every SPDLOG_* macro below this level, whatever the
# runtime level is later set to. Left at the default (INFO), `log_level:
# debug` silently changed nothing. Set here rather than in a header so it
# cannot depend on include order: PUBLIC, so it reaches mc_tracking, the
# tools and the tests too.
target_compile_definitions(mc_tracking_core
    PUBLIC SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE)

target_link_libraries(mc_tracking_core
    PUBLIC
        opencv_core
        opencv_imgproc
        Eigen3::Eigen
        spdlog::spdlog
        yaml-cpp::yaml-cpp
)
set_target_properties(mc_tracking_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---- Library: mc_tracking (static, GPU stages) ----
if(NOT MCT_CPU_ONLY)
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

# Already compiled into mc_tracking_core.
foreach(core_src
        system_config logger kalman_filter iou_tracker bytetrack
        hungarian identity_matcher reid_gallery tracker_factory)
    list(FILTER MCT_SOURCES EXCLUDE REGEX "${core_src}\\.cpp$")
endforeach()

add_library(mc_tracking STATIC ${MCT_SOURCES})
target_include_directories(mc_tracking
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        ${OpenCV_INCLUDE_DIRS}
        ${TensorRT_INCLUDE_DIRS}
)
target_link_libraries(mc_tracking
    PUBLIC
        mc_tracking_core
        CUDA::cudart
        CUDA::cuda_driver
        ${OpenCV_LIBS}
        Eigen3::Eigen
        spdlog::spdlog
        yaml-cpp::yaml-cpp
        TensorRT::nvinfer
        TensorRT::nvonnxparser
)
set_target_properties(mc_tracking PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---- Executable: mc_tracking_video (offline OpenCV driver) ----
add_executable(mc_tracking_video src/main.cpp)
target_link_libraries(mc_tracking_video PRIVATE mc_tracking)
endif()

if(BUILD_TOOLS)
    add_executable(mc_tracking_benchmark tools/benchmark.cpp)
    target_link_libraries(mc_tracking_benchmark PRIVATE mc_tracking)

    # Cross-camera ReID demo: splits one clip into two views and runs
    # the full MultiCameraOrchestrator over them.
    add_executable(crosscam_split_demo tools/crosscam_split_demo.cpp)
    target_link_libraries(crosscam_split_demo PRIVATE mc_tracking)
endif()

if(MCT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(TARGET mc_tracking_video)
    install(TARGETS mc_tracking_video RUNTIME DESTINATION bin)
endif()
if(BUILD_TOOLS)
    install(TARGETS mc_tracking_benchmark RUNTIME DESTINATION bin)
endif()
install(DIRECTORY configs/ DESTINATION share/mc_tracking/configs)

message(STATUS "")
message(STATUS "mc_tracking ${PROJECT_VERSION} configuration:")
message(STATUS "  Build type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard:     ${CMAKE_CXX_STANDARD}")
message(STATUS "  CPU-only:         ${MCT_CPU_ONLY}")
if(NOT MCT_CPU_ONLY)
    message(STATUS "  CUDA:             ${CUDAToolkit_VERSION}")
    message(STATUS "  TensorRT:         ${TensorRT_VERSION}")
endif()
message(STATUS "  OpenCV:           ${OpenCV_VERSION}")
message(STATUS "  Build tools:      ${BUILD_TOOLS}")
message(STATUS "  Build tests:      ${MCT_BUILD_TESTS}")
message(STATUS "")
