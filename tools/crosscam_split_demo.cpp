// crosscam_split_demo: a self-contained cross-camera ReID demo on a
// single clip.
//
// Splits each frame of one video into a LEFT and a RIGHT view - two
// adjacent, non-overlapping fields of view of the same walkway - and
// runs them through the full MultiCameraOrchestrator (per-view YOLO
// detect + BYTETrack + OSNet ReID, then the IdentityMatcher). Boxes
// are coloured by GLOBAL id, so a person leaving the left view and
// re-entering the right view keeps the same colour/id: that hand-off
// is exactly what the ReID + Hungarian matcher exists to do.
//
//   crosscam_split_demo --config configs/demo_crosscam.yaml \
//       --cameras configs/demo_crosscam_cameras.yaml \
//       --input walk.mp4 --output crosscam.mp4

#include <spdlog/spdlog.h>

#include <chrono>
#include <cstdlib>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <vector>

#include "mc_tracking/config/system_config.hpp"
#include "mc_tracking/overlay/visualizer.hpp"
#include "mc_tracking/pipeline/multi_camera.hpp"
#include "mc_tracking/utils/logger.hpp"

int main(int argc, char** argv) {
    std::string config_path, cameras_path, input, output;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto take = [&](const std::string& flag) {
            if (i + 1 >= argc) throw std::invalid_argument(flag + " expects a value");
            return std::string(argv[++i]);
        };
        if (a == "--config")
            config_path = take(a);
        else if (a == "--cameras")
            cameras_path = take(a);
        else if (a == "--input")
            input = take(a);
        else if (a == "--output")
            output = take(a);
        else if (a == "--help" || a == "-h") {
            std::cerr << "Usage: " << argv[0]
                      << " --config CFG --cameras CAMS --input VIDEO --output OUT\n";
            return EXIT_SUCCESS;
        }
    }
    if (config_path.empty() || cameras_path.empty() || input.empty() || output.empty()) {
        std::cerr << "missing required argument; see --help\n";
        return EXIT_FAILURE;
    }

    try {
        const auto cfg = mc_tracking::config::SystemConfig::load(config_path);
        const auto cams = mc_tracking::config::CamerasConfig::load(cameras_path);
        mc_tracking::utils::init_logger(cfg.logging.level, cfg.logging.json);

        mc_tracking::pipeline::MultiCameraOrchestrator orch(cfg, cams);
        auto& camL = orch.add_camera("cam-01", "left");
        auto& camR = orch.add_camera("cam-02", "right");
        mc_tracking::overlay::Visualizer viz;

        cv::VideoCapture cap(input);
        if (!cap.isOpened()) throw std::runtime_error("cannot open input: " + input);
        const int W = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        const int H = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        const double fps = cap.get(cv::CAP_PROP_FPS);
        const int halfW = W / 2;
        const int divider = 8;
        const int outW = W + divider;

        cv::VideoWriter writer(output, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                               (fps > 0.0 ? fps : 25.0), cv::Size(outW, H));
        if (!writer.isOpened()) throw std::runtime_error("cannot open output: " + output);

        cv::Mat frame, canvas(H, outW, CV_8UC3);
        std::uint64_t n = 0;
        while (cap.read(frame)) {
            const auto pts = std::chrono::steady_clock::now();
            cv::Mat left = frame(cv::Rect(0, 0, halfW, H)).clone();
            cv::Mat right = frame(cv::Rect(halfW, 0, W - halfW, H)).clone();

            auto rL = camL.process_frame(n, pts, left);
            auto rR = camR.process_frame(n, pts, right);
            std::vector<mc_tracking::pipeline::CameraFrameResult> results{rL, rR};
            orch.stamp_global_ids(results);

            viz.render(left, results[0]);
            viz.render(right, results[1]);

            canvas.setTo(cv::Scalar(40, 40, 40));
            left.copyTo(canvas(cv::Rect(0, 0, halfW, H)));
            right.copyTo(canvas(cv::Rect(halfW + divider, 0, W - halfW, H)));
            cv::line(canvas, cv::Point(halfW + divider / 2, 0), cv::Point(halfW + divider / 2, H),
                     cv::Scalar(80, 80, 80), 1);
            cv::putText(canvas, "cam-01 (left)", cv::Point(8, H - 12), cv::FONT_HERSHEY_SIMPLEX,
                        0.5, cv::Scalar(230, 230, 230), 1);
            cv::putText(canvas, "cam-02 (right)", cv::Point(halfW + divider + 8, H - 12),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(230, 230, 230), 1);
            writer.write(canvas);
            ++n;
        }
        SPDLOG_INFO("done: {} frames; same global id across views = re-identified person", n);
    } catch (const std::exception& e) {
        SPDLOG_CRITICAL("fatal: {}", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
