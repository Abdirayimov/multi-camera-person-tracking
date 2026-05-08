find_path(TensorRT_INCLUDE_DIR
    NAMES NvInfer.h
    HINTS
        /usr/include/x86_64-linux-gnu
        /usr/local/include
        ${TensorRT_ROOT}/include
)

find_library(TensorRT_NVINFER_LIB
    NAMES nvinfer
    HINTS
        /usr/lib/x86_64-linux-gnu
        /usr/local/lib
        ${TensorRT_ROOT}/lib
)

find_library(TensorRT_NVONNXPARSER_LIB
    NAMES nvonnxparser
    HINTS
        /usr/lib/x86_64-linux-gnu
        /usr/local/lib
        ${TensorRT_ROOT}/lib
)

if(TensorRT_INCLUDE_DIR AND EXISTS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")
    file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" _trt_major REGEX "#define NV_TENSORRT_MAJOR")
    file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" _trt_minor REGEX "#define NV_TENSORRT_MINOR")
    file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" _trt_patch REGEX "#define NV_TENSORRT_PATCH")
    string(REGEX REPLACE ".*#define NV_TENSORRT_MAJOR ([0-9]+).*" "\\1" _trt_major "${_trt_major}")
    string(REGEX REPLACE ".*#define NV_TENSORRT_MINOR ([0-9]+).*" "\\1" _trt_minor "${_trt_minor}")
    string(REGEX REPLACE ".*#define NV_TENSORRT_PATCH ([0-9]+).*" "\\1" _trt_patch "${_trt_patch}")
    set(TensorRT_VERSION "${_trt_major}.${_trt_minor}.${_trt_patch}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
    REQUIRED_VARS TensorRT_INCLUDE_DIR TensorRT_NVINFER_LIB TensorRT_NVONNXPARSER_LIB
    VERSION_VAR TensorRT_VERSION
)

if(TensorRT_FOUND)
    set(TensorRT_INCLUDE_DIRS "${TensorRT_INCLUDE_DIR}")
    if(NOT TARGET TensorRT::nvinfer)
        add_library(TensorRT::nvinfer UNKNOWN IMPORTED)
        set_target_properties(TensorRT::nvinfer PROPERTIES
            IMPORTED_LOCATION "${TensorRT_NVINFER_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}")
    endif()
    if(NOT TARGET TensorRT::nvonnxparser)
        add_library(TensorRT::nvonnxparser UNKNOWN IMPORTED)
        set_target_properties(TensorRT::nvonnxparser PROPERTIES
            IMPORTED_LOCATION "${TensorRT_NVONNXPARSER_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}")
    endif()
endif()

mark_as_advanced(TensorRT_INCLUDE_DIR TensorRT_NVINFER_LIB TensorRT_NVONNXPARSER_LIB)
