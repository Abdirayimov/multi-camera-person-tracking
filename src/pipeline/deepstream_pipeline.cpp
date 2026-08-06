#include "mc_tracking/pipeline/deepstream_pipeline.hpp"

#include <gst/gst.h>
#include <gstnvdsmeta.h>
#include <spdlog/spdlog.h>

#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace mc_tracking::pipeline {

namespace {

constexpr int kMaxSourcesHardLimit = 32;

/// Context handed to the per-source pad-added callback: nvurisrcbin
/// exposes its video pad dynamically, so the link to the muxer's request
/// pad has to happen when the pad appears, not at add_source time.
struct SourceLink {
    GstPad* mux_pad = nullptr;  ///< owned; released in ~Impl
    std::string camera_id;
};

}  // namespace

struct DeepStreamPipeline::Impl {
    GstElement* pipeline = nullptr;
    GstElement* streammux = nullptr;
    GstElement* pgie = nullptr;
    GstElement* tracker = nullptr;
    GstElement* sink = nullptr;
    GMainLoop* loop = nullptr;
    std::thread loop_thread;
    guint bus_watch_id = 0;

    TrackBatchCallback callback;

    /// pad index -> camera id, fixed before start(); the probe reads it
    /// from a streaming thread without a lock, which is only safe because
    /// add_source refuses to run once the pipeline is started.
    std::vector<std::string> camera_by_pad;

    std::vector<std::unique_ptr<SourceLink>> links;

    static GstPadProbeReturn on_tracker_buffer(GstPad* /*pad*/, GstPadProbeInfo* info,
                                               gpointer user_data);
    static void on_src_pad_added(GstElement* src, GstPad* new_pad, gpointer user_data);
    static gboolean on_bus_message(GstBus* bus, GstMessage* msg, gpointer user_data);
};

// This is the wiring that turns nvtracker output into library types: every
// NvDsObjectMeta after nvtracker carries the NvDCF-assigned object_id,
// stable per stream across occlusions.
GstPadProbeReturn DeepStreamPipeline::Impl::on_tracker_buffer(GstPad* /*pad*/,
                                                              GstPadProbeInfo* info,
                                                              gpointer user_data) {
    auto* impl = static_cast<Impl*>(user_data);
    if (!impl->callback) {
        return GST_PAD_PROBE_OK;
    }
    GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);
    const NvDsBatchMeta* batch = gst_buffer_get_nvds_batch_meta(buf);
    if (batch == nullptr) {
        return GST_PAD_PROBE_OK;
    }

    for (NvDsMetaList* lf = batch->frame_meta_list; lf != nullptr; lf = lf->next) {
        auto* fm = static_cast<NvDsFrameMeta*>(lf->data);

        CameraFrameResult out;
        const auto pad = static_cast<std::size_t>(fm->pad_index);
        out.camera_id = pad < impl->camera_by_pad.size() ? impl->camera_by_pad[pad] : "unknown";
        out.frame_number = static_cast<std::uint64_t>(fm->frame_num);
        out.pts = std::chrono::steady_clock::now();

        for (NvDsMetaList* lo = fm->obj_meta_list; lo != nullptr; lo = lo->next) {
            auto* om = static_cast<NvDsObjectMeta*>(lo->data);
            tracker::Track t;
            t.local_id = om->object_id;
            t.camera_id = out.camera_id;
            t.bbox = cv::Rect2f(static_cast<float>(om->rect_params.left),
                                static_cast<float>(om->rect_params.top),
                                static_cast<float>(om->rect_params.width),
                                static_cast<float>(om->rect_params.height));
            t.confidence = om->confidence;
            // NvDCF handles its own probation internally; everything it
            // emits downstream is an active track.
            t.state = tracker::TrackState::Confirmed;
            out.tracks.push_back(std::move(t));
        }
        impl->callback(out);
    }
    return GST_PAD_PROBE_OK;
}

void DeepStreamPipeline::Impl::on_src_pad_added(GstElement* /*src*/, GstPad* new_pad,
                                                gpointer user_data) {
    auto* link = static_cast<SourceLink*>(user_data);
    if (gst_pad_is_linked(link->mux_pad)) {
        return;
    }
    // nvurisrcbin also announces audio pads; only the NVMM video pad
    // belongs on the muxer.
    GstCaps* caps = gst_pad_get_current_caps(new_pad);
    if (caps == nullptr) {
        caps = gst_pad_query_caps(new_pad, nullptr);
    }
    const GstStructure* s = gst_caps_get_structure(caps, 0);
    const bool is_video = g_str_has_prefix(gst_structure_get_name(s), "video/");
    gst_caps_unref(caps);
    if (!is_video) {
        return;
    }
    if (gst_pad_link(new_pad, link->mux_pad) != GST_PAD_LINK_OK) {
        SPDLOG_ERROR("[{}] failed to link source pad to muxer", link->camera_id);
    } else {
        SPDLOG_INFO("[{}] source pad linked", link->camera_id);
    }
}

gboolean DeepStreamPipeline::Impl::on_bus_message(GstBus* /*bus*/, GstMessage* msg,
                                                  gpointer user_data) {
    auto* impl = static_cast<Impl*>(user_data);
    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_ERROR: {
            GError* err = nullptr;
            gchar* dbg = nullptr;
            gst_message_parse_error(msg, &err, &dbg);
            SPDLOG_ERROR("GStreamer error from {}: {} ({})", GST_OBJECT_NAME(msg->src),
                         err->message, dbg ? dbg : "no detail");
            g_clear_error(&err);
            g_free(dbg);
            if (impl->loop) {
                g_main_loop_quit(impl->loop);
            }
            break;
        }
        case GST_MESSAGE_EOS:
            SPDLOG_INFO("EOS on every source; stopping");
            if (impl->loop) {
                g_main_loop_quit(impl->loop);
            }
            break;
        default:
            break;
    }
    return TRUE;
}

DeepStreamPipeline::DeepStreamPipeline(const config::SystemConfig& cfg,
                                       const std::string& pgie_config_path)
    : impl_(std::make_unique<Impl>()), cfg_(cfg) {
    if (!gst_is_initialized()) {
        gst_init(nullptr, nullptr);
    }

    impl_->pipeline = gst_pipeline_new("mc-tracking-ds");
    impl_->streammux = gst_element_factory_make("nvstreammux", "stream-muxer");
    impl_->pgie = gst_element_factory_make("nvinfer", "primary-detector");
    impl_->tracker = gst_element_factory_make("nvtracker", "nvdcf-tracker");
    impl_->sink = gst_element_factory_make("fakesink", "sink");

    if (!impl_->pipeline || !impl_->streammux || !impl_->pgie || !impl_->tracker || !impl_->sink) {
        throw std::runtime_error(
            "failed to create a GStreamer element; is the DeepStream runtime installed?");
    }

    g_object_set(G_OBJECT(impl_->streammux), "width", cfg_.pipeline.muxer_width, "height",
                 cfg_.pipeline.muxer_height, "batched-push-timeout", 40000, nullptr);

    g_object_set(G_OBJECT(impl_->pgie), "config-file-path", pgie_config_path.c_str(), nullptr);

    const auto& nv = cfg_.tracker.nvdcf;
    g_object_set(G_OBJECT(impl_->tracker), "tracker-width", nv.tracker_width, "tracker-height",
                 nv.tracker_height, "ll-lib-file",
                 "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
                 "ll-config-file", nv.config_path.c_str(), nullptr);

    g_object_set(G_OBJECT(impl_->sink), "sync", FALSE, "async", FALSE, "qos", FALSE, nullptr);

    gst_bin_add_many(GST_BIN(impl_->pipeline), impl_->streammux, impl_->pgie, impl_->tracker,
                     impl_->sink, nullptr);
    if (!gst_element_link_many(impl_->streammux, impl_->pgie, impl_->tracker, impl_->sink,
                               nullptr)) {
        throw std::runtime_error("failed to link streammux -> pgie -> nvtracker -> sink");
    }

    // Attach the probe that hands nvtracker's output to the callback.
    GstPad* tracker_src = gst_element_get_static_pad(impl_->tracker, "src");
    if (tracker_src == nullptr) {
        throw std::runtime_error("nvtracker has no src pad");
    }
    gst_pad_add_probe(tracker_src, GST_PAD_PROBE_TYPE_BUFFER, &Impl::on_tracker_buffer, impl_.get(),
                      nullptr);
    gst_object_unref(tracker_src);

    GstBus* bus = gst_pipeline_get_bus(GST_PIPELINE(impl_->pipeline));
    impl_->bus_watch_id = gst_bus_add_watch(bus, &Impl::on_bus_message, impl_.get());
    gst_object_unref(bus);
}

DeepStreamPipeline::~DeepStreamPipeline() {
    stop();
    if (impl_->loop_thread.joinable()) {
        impl_->loop_thread.join();
    }
    if (impl_->bus_watch_id != 0) {
        g_source_remove(impl_->bus_watch_id);
    }
    for (auto& link : impl_->links) {
        if (link->mux_pad != nullptr) {
            gst_object_unref(link->mux_pad);
        }
    }
    if (impl_->pipeline) {
        gst_element_set_state(impl_->pipeline, GST_STATE_NULL);
        gst_object_unref(impl_->pipeline);
    }
}

bool DeepStreamPipeline::add_source(const std::string& camera_id, const std::string& uri) {
    std::lock_guard<std::mutex> lock(sources_mutex_);
    if (running_) {
        SPDLOG_WARN("add_source rejected: pipeline already running");
        return false;
    }
    if (sources_.size() >= static_cast<std::size_t>(kMaxSourcesHardLimit)) {
        SPDLOG_WARN("add_source rejected: hard source limit ({})", kMaxSourcesHardLimit);
        return false;
    }
    if (sources_.count(camera_id) != 0) {
        SPDLOG_WARN("add_source rejected: duplicate camera_id '{}'", camera_id);
        return false;
    }

    const int idx = static_cast<int>(sources_.size());
    const std::string bin_name = "src-bin-" + std::to_string(idx);

    GstElement* bin = gst_element_factory_make("nvurisrcbin", bin_name.c_str());
    if (bin == nullptr) {
        SPDLOG_ERROR("nvurisrcbin creation failed for '{}'", camera_id);
        return false;
    }
    g_object_set(G_OBJECT(bin), "uri", uri.c_str(), nullptr);
    if (!gst_bin_add(GST_BIN(impl_->pipeline), bin)) {
        gst_object_unref(bin);
        return false;
    }

    const std::string pad_name = "sink_" + std::to_string(idx);
    GstPad* mux_pad = gst_element_request_pad_simple(impl_->streammux, pad_name.c_str());
    if (mux_pad == nullptr) {
        SPDLOG_ERROR("muxer refused pad '{}' for '{}'", pad_name, camera_id);
        return false;
    }

    auto link = std::make_unique<SourceLink>();
    link->mux_pad = mux_pad;
    link->camera_id = camera_id;
    g_signal_connect(bin, "pad-added", G_CALLBACK(&Impl::on_src_pad_added), link.get());
    impl_->links.push_back(std::move(link));

    impl_->camera_by_pad.push_back(camera_id);
    sources_[camera_id] = idx;
    SPDLOG_INFO("added source '{}' as pad {}", camera_id, idx);
    return true;
}

void DeepStreamPipeline::set_track_callback(TrackBatchCallback cb) {
    impl_->callback = std::move(cb);
}

void DeepStreamPipeline::start() {
    if (running_.exchange(true)) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(sources_mutex_);
        const auto n = static_cast<guint>(sources_.size());
        // The muxer batches one frame per source; the detector must accept
        // the same batch. An engine built for a smaller max batch will make
        // nvinfer fail loudly here, which beats silently dropping sources.
        g_object_set(G_OBJECT(impl_->streammux), "batch-size", n, nullptr);
        g_object_set(G_OBJECT(impl_->pgie), "batch-size", n, nullptr);
    }
    impl_->loop = g_main_loop_new(nullptr, FALSE);
    impl_->loop_thread = std::thread([this]() {
        gst_element_set_state(impl_->pipeline, GST_STATE_PLAYING);
        g_main_loop_run(impl_->loop);
        gst_element_set_state(impl_->pipeline, GST_STATE_NULL);
    });
    SPDLOG_INFO("DeepStream pipeline started");
}

void DeepStreamPipeline::wait() {
    if (impl_->loop_thread.joinable()) {
        impl_->loop_thread.join();
    }
    running_ = false;
}

void DeepStreamPipeline::stop() {
    if (!running_.exchange(false)) {
        return;
    }
    if (impl_->loop) {
        g_main_loop_quit(impl_->loop);
    }
}

}  // namespace mc_tracking::pipeline
