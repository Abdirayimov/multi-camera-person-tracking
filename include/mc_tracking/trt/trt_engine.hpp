#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace nvinfer1 {
class ICudaEngine;
class IExecutionContext;
class IRuntime;
}  // namespace nvinfer1

namespace mc_tracking::trt {

struct BindingInfo {
    std::string name;
    std::vector<std::int64_t> shape;
    std::size_t element_size = 0;
    std::size_t volume = 0;
    bool is_input = false;
};

class TrtEngine {
public:
    explicit TrtEngine(const std::string& engine_path);
    ~TrtEngine();

    TrtEngine(const TrtEngine&) = delete;
    TrtEngine& operator=(const TrtEngine&) = delete;
    TrtEngine(TrtEngine&&) noexcept;
    TrtEngine& operator=(TrtEngine&&) noexcept;

    void set_input_shape(const std::string& name, const std::vector<std::int64_t>& shape);
    void copy_input(const std::string& name, const void* host_src, std::size_t bytes,
                    cudaStream_t stream);
    void copy_output(const std::string& name, void* host_dst, std::size_t bytes,
                     cudaStream_t stream) const;
    void infer(cudaStream_t stream);

    void* device_ptr(const std::string& name);
    const void* device_ptr(const std::string& name) const;

    const std::vector<BindingInfo>& bindings() const noexcept { return bindings_; }
    const BindingInfo& binding(const std::string& name) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::vector<BindingInfo> bindings_;
};

}  // namespace mc_tracking::trt
