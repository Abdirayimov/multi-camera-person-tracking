set(_DS_ROOTS
    /opt/nvidia/deepstream/deepstream
    /opt/nvidia/deepstream/deepstream-8.0
    /opt/nvidia/deepstream/deepstream-7.1
    ${DeepStream_ROOT}
)

find_path(DeepStream_INCLUDE_DIR
    NAMES nvds_meta.h
    HINTS ${_DS_ROOTS}
    PATH_SUFFIXES sources/includes
)

find_library(DeepStream_NVDS_META_LIB NAMES nvds_meta
    HINTS ${_DS_ROOTS} PATH_SUFFIXES lib)
find_library(DeepStream_NVBUFSURFACE_LIB NAMES nvbufsurface
    HINTS ${_DS_ROOTS} PATH_SUFFIXES lib)

if(DeepStream_INCLUDE_DIR)
    string(REGEX MATCH "deepstream-([0-9]+\\.[0-9]+)" _ds_match "${DeepStream_INCLUDE_DIR}")
    if(_ds_match)
        set(DeepStream_VERSION "${CMAKE_MATCH_1}")
    else()
        set(DeepStream_VERSION "unknown")
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(DeepStream
    REQUIRED_VARS DeepStream_INCLUDE_DIR DeepStream_NVDS_META_LIB DeepStream_NVBUFSURFACE_LIB
    VERSION_VAR DeepStream_VERSION
)

if(DeepStream_FOUND)
    set(DeepStream_INCLUDE_DIRS "${DeepStream_INCLUDE_DIR}")
    if(NOT TARGET DeepStream::nvds_meta)
        add_library(DeepStream::nvds_meta UNKNOWN IMPORTED)
        set_target_properties(DeepStream::nvds_meta PROPERTIES
            IMPORTED_LOCATION "${DeepStream_NVDS_META_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${DeepStream_INCLUDE_DIR}")
    endif()
    if(NOT TARGET DeepStream::nvbufsurface)
        add_library(DeepStream::nvbufsurface UNKNOWN IMPORTED)
        set_target_properties(DeepStream::nvbufsurface PROPERTIES
            IMPORTED_LOCATION "${DeepStream_NVBUFSURFACE_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${DeepStream_INCLUDE_DIR}")
    endif()
endif()

mark_as_advanced(DeepStream_INCLUDE_DIR DeepStream_NVDS_META_LIB DeepStream_NVBUFSURFACE_LIB)
