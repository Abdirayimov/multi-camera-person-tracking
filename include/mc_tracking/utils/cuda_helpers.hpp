#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace mc_tracking::utils {

inline void cuda_check(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error at ") + file + ":" +
                                 std::to_string(line) + " - " + cudaGetErrorString(err));
    }
}

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
