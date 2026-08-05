#include "mc_tracking/trt/trt_engine.hpp"

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mc_tracking/utils/cuda_helpers.hpp"

namespace mc_tracking::trt {

namespace {

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
            case Severity::kERROR:
                SPDLOG_ERROR("[TRT] {}", msg);
                break;
            case Severity::kWARNING:
                SPDLOG_WARN("[TRT] {}", msg);
                break;
            case Severity::kINFO:
                SPDLOG_DEBUG("[TRT] {}", msg);
                break;
            case Severity::kVERBOSE:
                SPDLOG_TRACE("[TRT] {}", msg);
                break;
        }
    }
};

TrtLogger& global_trt_logger() {
    static TrtLogger inst;
    return inst;
}

std::size_t element_size_for(nvinfer1::DataType dt) {
    switch (dt) {
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kINT8:
            return 1;
        case nvinfer1::DataType::kINT32:
            return 4;
        case nvinfer1::DataType::kBOOL:
            return 1;
        case nvinfer1::DataType::kUINT8:
            return 1;
        case nvinfer1::DataType::kFP8:
            return 1;
        default:
            return 0;
    }
}

std::size_t volume_of(const std::vector<std::int64_t>& s) {
    std::size_t v = 1;
    for (auto d : s) {
        if (d <= 0) return 0;
        v *= static_cast<std::size_t>(d);
    }
    return v;
}

}  // namespace

struct TrtEngine::Impl {
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    // Owning. The constructor allocates one of these per binding and can
    // throw part-way through; DeviceBuffer frees what was already taken,
    // which ~TrtEngine cannot do because it never runs for an object whose
    // constructor threw.
    std::unordered_map<std::string, utils::DeviceBuffer> device_buffers;
};

TrtEngine::TrtEngine(const std::string& engine_path) : impl_(std::make_unique<Impl>()) {
    std::ifstream f(engine_path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) throw std::runtime_error("cannot open TRT engine: " + engine_path);
    const std::streamsize sz = f.tellg();
    // tellg returns -1 on failure, which would size the vector from a huge
    // unsigned value; an empty file is equally not an engine.
    if (sz <= 0) throw std::runtime_error("TRT engine file is empty or unreadable: " + engine_path);
    f.seekg(0, std::ios::beg);
    std::vector<char> blob(static_cast<std::size_t>(sz));
    if (!f.read(blob.data(), sz)) {
        throw std::runtime_error("short read on TRT engine: " + engine_path);
    }

    impl_->runtime.reset(nvinfer1::createInferRuntime(global_trt_logger()));
    if (!impl_->runtime) throw std::runtime_error("createInferRuntime failed");
    impl_->engine.reset(impl_->runtime->deserializeCudaEngine(blob.data(), blob.size()));
    if (!impl_->engine)
        throw std::runtime_error("failed to deserialize TRT engine: " + engine_path);
    impl_->context.reset(impl_->engine->createExecutionContext());
    if (!impl_->context) throw std::runtime_error("failed to create TRT execution context");

    const std::int32_t n = impl_->engine->getNbIOTensors();
    bindings_.reserve(static_cast<std::size_t>(n));
    for (std::int32_t i = 0; i < n; ++i) {
        const char* name = impl_->engine->getIOTensorName(i);
        const auto dims = impl_->engine->getTensorShape(name);
        const auto dtype = impl_->engine->getTensorDataType(name);
        const auto io = impl_->engine->getTensorIOMode(name);

        BindingInfo info;
        info.name = name;
        info.shape.reserve(static_cast<std::size_t>(dims.nbDims));
        for (std::int32_t k = 0; k < dims.nbDims; ++k) {
            info.shape.push_back(dims.d[k]);
        }
        info.element_size = element_size_for(dtype);
        info.volume = volume_of(info.shape);
        info.is_input = (io == nvinfer1::TensorIOMode::kINPUT);

        if (info.volume > 0 && info.element_size > 0) {
            utils::DeviceBuffer buf(info.volume * info.element_size);
            impl_->context->setTensorAddress(name, buf.get());
            impl_->device_buffers[info.name] = std::move(buf);
        }
        bindings_.push_back(std::move(info));
    }
}

// Defined here, where Impl is complete, and left in place: pimpl needs the
// destructor in this translation unit even though DeviceBuffer now does the
// freeing.
TrtEngine::~TrtEngine() = default;

TrtEngine::TrtEngine(TrtEngine&&) noexcept = default;
TrtEngine& TrtEngine::operator=(TrtEngine&&) noexcept = default;

void TrtEngine::set_input_shape(const std::string& name, const std::vector<std::int64_t>& shape) {
    // nvinfer1::Dims::d is a fixed array of MAX_DIMS; writing past it
    // corrupts the stack frame this Dims sits in.
    if (shape.size() > static_cast<std::size_t>(nvinfer1::Dims::MAX_DIMS)) {
        throw std::invalid_argument("shape rank " + std::to_string(shape.size()) + " for '" + name +
                                    "' exceeds nvinfer1::Dims::MAX_DIMS (" +
                                    std::to_string(nvinfer1::Dims::MAX_DIMS) + ")");
    }
    nvinfer1::Dims dims;
    dims.nbDims = static_cast<std::int32_t>(shape.size());
    for (std::size_t i = 0; i < shape.size(); ++i) dims.d[i] = shape[i];
    if (!impl_->context->setInputShape(name.c_str(), dims)) {
        throw std::runtime_error("setInputShape failed for " + name);
    }

    // Allocate before touching the binding metadata. reset() throwing used
    // to leave info.shape describing the new shape while the buffer was
    // gone, so the next copy_input wrote into a null device pointer.
    auto& info = mutable_binding(name);
    auto& buf = impl_->device_buffers[name];
    buf.reset(volume_of(shape) * info.element_size);
    info.shape = shape;
    info.volume = volume_of(shape);
    impl_->context->setTensorAddress(name.c_str(), buf.get());

    // Setting an input shape resolves previously-dynamic output shapes.
    // Re-query each output binding from the context and (re)allocate +
    // bind its device buffer, otherwise a dynamic-batch engine (e.g. the
    // OSNet ReID extractor) has no address set for its outputs and
    // enqueueV3 fails.
    for (auto& out : bindings_) {
        if (out.is_input) continue;
        const auto odims = impl_->context->getTensorShape(out.name.c_str());
        if (odims.nbDims < 0) continue;
        std::vector<std::int64_t> resolved;
        bool concrete = true;
        for (std::int32_t k = 0; k < odims.nbDims; ++k) {
            resolved.push_back(odims.d[k]);
            if (odims.d[k] < 0) concrete = false;
        }
        if (!concrete) continue;
        const std::size_t new_vol = volume_of(resolved);
        auto& obuf = impl_->device_buffers[out.name];
        if (new_vol == out.volume && obuf.get() != nullptr) continue;
        obuf.reset(new_vol * out.element_size);
        out.shape = resolved;
        out.volume = new_vol;
        impl_->context->setTensorAddress(out.name.c_str(), obuf.get());
    }
}

void TrtEngine::copy_input(const std::string& name, const void* host_src, std::size_t bytes,
                           cudaStream_t stream) {
    void* dst = impl_->device_buffers.at(name).get();
    MCT_CUDA_CHECK(cudaMemcpyAsync(dst, host_src, bytes, cudaMemcpyHostToDevice, stream));
}

void TrtEngine::copy_output(const std::string& name, void* host_dst, std::size_t bytes,
                            cudaStream_t stream) const {
    const void* src = impl_->device_buffers.at(name).get();
    MCT_CUDA_CHECK(cudaMemcpyAsync(host_dst, src, bytes, cudaMemcpyDeviceToHost, stream));
}

void TrtEngine::infer(cudaStream_t stream) {
    if (!impl_->context->enqueueV3(stream)) {
        throw std::runtime_error("enqueueV3 failed");
    }
}

void* TrtEngine::device_ptr(const std::string& name) {
    return impl_->device_buffers.at(name).get();
}

const void* TrtEngine::device_ptr(const std::string& name) const {
    return impl_->device_buffers.at(name).get();
}

const BindingInfo& TrtEngine::binding(const std::string& name) const {
    for (const auto& b : bindings_) {
        if (b.name == name) return b;
    }
    throw std::out_of_range("no such binding: " + name);
}

BindingInfo& TrtEngine::mutable_binding(const std::string& name) {
    // Indexed rather than range-based: a `auto& b` loop that returns b trips
    // cppcheck's constVariableReference, and the const-ref form it suggests
    // would not compile against this return type.
    for (std::size_t i = 0; i < bindings_.size(); ++i) {
        if (bindings_[i].name == name) return bindings_[i];
    }
    throw std::out_of_range("no such binding: " + name);
}

}  // namespace mc_tracking::trt
