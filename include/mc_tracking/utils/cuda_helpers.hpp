#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace mc_tracking::utils {

inline void cuda_check(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error at ") + file + ":" + std::to_string(line) +
                                 " - " + cudaGetErrorString(err));
    }
}

/// Owning wrapper over a single `cudaMalloc` allocation.
///
/// Exists so that a constructor which allocates several buffers and then
/// throws does not leak the ones it already got: a partially-constructed
/// object never runs its destructor, but its fully-constructed members do.
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
        if (bytes > 0) cuda_check(cudaMalloc(&data_, bytes), __FILE__, __LINE__);
    }
    ~DeviceBuffer() { release(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept : data_(other.data_), bytes_(other.bytes_) {
        other.data_ = nullptr;
        other.bytes_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            data_ = other.data_;
            bytes_ = other.bytes_;
            other.data_ = nullptr;
            other.bytes_ = 0;
        }
        return *this;
    }

    /// Free the current allocation. Never throws, so the destructor and the
    /// move assignment can call it without risking std::terminate.
    void release() noexcept {
        if (data_ != nullptr) {
            cudaFree(data_);
            data_ = nullptr;
            bytes_ = 0;
        }
    }

    /// Free the current allocation and, if `new_bytes > 0`, allocate a new
    /// one. Throws on allocation failure, leaving the buffer empty.
    void reset(std::size_t new_bytes = 0) {
        release();
        if (new_bytes > 0) {
            cuda_check(cudaMalloc(&data_, new_bytes), __FILE__, __LINE__);
            bytes_ = new_bytes;
        }
    }

    void* get() noexcept { return data_; }
    const void* get() const noexcept { return data_; }
    std::size_t bytes() const noexcept { return bytes_; }

private:
    void* data_ = nullptr;
    std::size_t bytes_ = 0;
};

class CudaStream {
public:
    CudaStream() { cuda_check(cudaStreamCreate(&stream_), __FILE__, __LINE__); }
    ~CudaStream() {
        if (stream_ != nullptr) cudaStreamDestroy(stream_);
    }
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) { other.stream_ = nullptr; }
    CudaStream& operator=(CudaStream&& other) noexcept {
        if (this != &other) {
            if (stream_ != nullptr) cudaStreamDestroy(stream_);
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }
    cudaStream_t get() const noexcept { return stream_; }
    void synchronize() const { cuda_check(cudaStreamSynchronize(stream_), __FILE__, __LINE__); }

private:
    cudaStream_t stream_ = nullptr;
};

}  // namespace mc_tracking::utils

#define MCT_CUDA_CHECK(call) ::mc_tracking::utils::cuda_check((call), __FILE__, __LINE__)
