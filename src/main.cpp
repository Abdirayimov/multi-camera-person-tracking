// mc_tracking_video: OpenCV-based driver for single-source or
// multi-source offline processing.
//
// Modes:
//   --input ONE.mp4 --output ONE.mp4
//       Single camera. Uses the system tracker / reid config.
//
//   --cameras cameras.yaml --output-dir DIR
//       Multi-camera. One mp4 per camera in cameras.yaml; the
//       orchestrator runs cross-camera matching across them.

#include <spdlog/spdlog.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <string>

#include "mc_tracking/config/system_config.hpp"
#include "mc_tracking/overlay/visualizer.hpp"
#include "mc_tracking/pipeline/multi_camera.hpp"
#include "mc_tracking/pipeline/single_camera.hpp"
#include "mc_tracking/utils/logger.hpp"

namespace {

std::atomic<bool> g_shutdown{false};
void sig(int) { g_shutdown = true; }

void print_usage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0 << " --config CONFIG_YAML\n"
        << "       (--input VIDEO --output ANNOTATED.mp4 |\n"
        << "        --cameras cameras.yaml --output-dir DIR)\n";
}

int run_single(const std::string& config_path, const std::string& input,
               const std::string& output) {
    using namespace mc_tracking;

    const auto cfg = config::SystemConfig::load(config_path);
    utils::init_logger(cfg.logging.level, cfg.logging.json);

    pipeline::SingleCameraPipeline pipe("cam-01", "single", cfg);
    overlay::Visualizer viz;

    cv::VideoCapture cap(input);
    if (!cap.isOpened()) throw std::runtime_error("could not open input video: " + input);
    const int W = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int H = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    const double fps = cap.get(cv::CAP_PROP_FPS);
    cv::VideoWriter writer(output, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                           (fps > 0.0 ? fps : 25.0), cv::Size(W, H));
    if (!writer.isOpened()) throw std::runtime_error("could not open output: " + output);

    std::uint64_t frame_no = 0;
    cv::Mat frame;
    while (cap.read(frame)) {
        if (g_shutdown.load()) break;
        const auto pts = std::chrono::steady_clock::now();
        const auto result = pipe.process_frame(frame_no++, pts, frame);
        viz.render(frame, result);
        writer.write(frame);
        if (frame_no % 30 == 0) {
            SPDLOG_INFO("frame={} tracks={}", frame_no, result.tracks.size());
        }
    }
    SPDLOG_INFO("done: {} frames processed", frame_no);
    return EXIT_SUCCESS;
}

int run_multi(const std::string& config_path, const std::string& cameras_path,
              const std::string& output_dir) {
    using namespace mc_tracking;

    const auto cfg = config::SystemConfig::load(config_path);
    const auto cams = config::CamerasConfig::load(cameras_path);
    utils::init_logger(cfg.logging.level, cfg.logging.json);

    pipeline::MultiCameraOrchestrator orch(cfg, cams);
    overlay::Visualizer viz;

    struct CamRunner {
        std::string id;
        std::string output;
        std::unique_ptr<cv::VideoCapture> cap;
        std::unique_ptr<cv::VideoWriter> writer;
        pipeline::SingleCameraPipeline* pipe = nullptr;
        std::uint64_t frame_no = 0;
    };
    std::vector<CamRunner> runners;
    for (const auto& cam : cams.cameras) {
        CamRunner r;
        r.id = cam.id;
        r.output = output_dir + "/" + cam.id + ".mp4";
        r.cap = std::make_unique<cv::VideoCapture>(cam.uri);
        if (!r.cap->isOpened()) {
            SPDLOG_WARN("could not open camera {} uri={}", cam.id, cam.uri);
            continue;
        }
        const int W = static_cast<int>(r.cap->get(cv::CAP_PROP_FRAME_WIDTH));
        const int H = static_cast<int>(r.cap->get(cv::CAP_PROP_FRAME_HEIGHT));
        const double fps = r.cap->get(cv::CAP_PROP_FPS);
        r.writer = std::make_unique<cv::VideoWriter>(
            r.output, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
            (fps > 0.0 ? fps : 25.0), cv::Size(W, H));
        r.pipe = &orch.add_camera(cam.id, cam.zone);
        runners.push_back(std::move(r));
    }
    SPDLOG_INFO("running {} camera(s); cross-cam matching {}", runners.size(),
                cfg.crosscam.enabled ? "ENABLED" : "DISABLED");

    while (!g_shutdown.load()) {
        std::vector<pipeline::CameraFrameResult> frame_results;
        std::vector<cv::Mat> live_frames(runners.size());
        bool any_active = false;

        for (std::size_t i = 0; i < runners.size(); ++i) {
            cv::Mat frame;
            if (!runners[i].cap->read(frame)) continue;
            any_active = true;
            const auto pts = std::chrono::steady_clock::now();
            auto r = runners[i].pipe->process_frame(runners[i].frame_no++, pts, frame);
            live_frames[i] = std::move(frame);
            frame_results.push_back(std::move(r));
        }
        if (!any_active) break;

        orch.stamp_global_ids(frame_results);

        // Render + write per-camera output.
        for (std::size_t k = 0; k < frame_results.size(); ++k) {
            // Find the matching runner by camera id.
            for (auto& r : runners) {
                if (r.id != frame_results[k].camera_id) continue;
                if (!live_frames[k].empty()) {
                    viz.render(live_frames[k], frame_results[k]);
                    r.writer->write(live_frames[k]);
                }
                break;
            }
        }
    }
    return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv) {
    std::string config_path;
    std::string input_path;
    std::string output_path;
    std::string cameras_path;
    std::string output_dir;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto take = [&](const std::string& flag) {
            if (i + 1 >= argc) throw std::invalid_argument(flag + " expects a value");
            return std::string(argv[++i]);
        };
        if (a == "--config" || a == "-c") config_path = take(a);
        else if (a == "--input" || a == "-i") input_path = take(a);
        else if (a == "--output" || a == "-o") output_path = take(a);
        else if (a == "--cameras") cameras_path = take(a);
        else if (a == "--output-dir") output_dir = take(a);
        else if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        }
    }
    if (config_path.empty()) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    std::signal(SIGINT, sig);
    std::signal(SIGTERM, sig);

    try {
        if (!input_path.empty() && !output_path.empty()) {
            return run_single(config_path, input_path, output_path);
        }
        if (!cameras_path.empty() && !output_dir.empty()) {
            return run_multi(config_path, cameras_path, output_dir);
        }
        print_usage(argv[0]);
        return EXIT_FAILURE;
    } catch (const std::exception& e) {
        SPDLOG_CRITICAL("fatal: {}", e.what());
        return EXIT_FAILURE;
    }
}
