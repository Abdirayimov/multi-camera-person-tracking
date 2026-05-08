// benchmark: per-stage latency for the multi-camera tracking pipeline
// using synthetic frames.

#include <spdlog/spdlog.h>

#include <opencv2/core.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "mc_tracking/config/system_config.hpp"
#include "mc_tracking/pipeline/single_camera.hpp"
#include "mc_tracking/utils/logger.hpp"

namespace {

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const auto pos = p * static_cast<double>(v.size() - 1);
    const auto lo = static_cast<std::size_t>(pos);
    const double frac = pos - static_cast<double>(lo);
    if (lo + 1 >= v.size()) return v[lo];
    return v[lo] * (1.0 - frac) + v[lo + 1] * frac;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " --config CONFIG_YAML\n";
        return EXIT_FAILURE;
    }
    std::string config_path;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if ((a == "--config" || a == "-c") && i + 1 < argc) config_path = argv[++i];
    }
    if (config_path.empty()) return EXIT_FAILURE;

    using namespace mc_tracking;

    const auto cfg = config::SystemConfig::load(config_path);
    utils::init_logger("warn", false);

    pipeline::SingleCameraPipeline pipe("bench", "synthetic", cfg);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> ud(0, 255);
    cv::Mat frame(720, 1280, CV_8UC3, cv::Scalar(64, 64, 64));
    for (int y = 0; y < frame.rows; ++y) {
        for (int x = 0; x < frame.cols; ++x) {
            auto& px = frame.at<cv::Vec3b>(y, x);
            px[0] = static_cast<unsigned char>(ud(rng));
            px[1] = static_cast<unsigned char>(ud(rng));
            px[2] = static_cast<unsigned char>(ud(rng));
        }
    }

    constexpr std::size_t warmup = 5;
    constexpr std::size_t iters = 100;
    using Clock = std::chrono::steady_clock;
    for (std::size_t i = 0; i < warmup; ++i) {
        (void)pipe.process_frame(i, Clock::now(), frame);
    }

    std::vector<double> ms;
    ms.reserve(iters);
    for (std::size_t i = 0; i < iters; ++i) {
        const auto t0 = Clock::now();
        (void)pipe.process_frame(i + warmup, Clock::now(), frame);
        const auto t1 = Clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    std::cout << std::fixed << std::setprecision(3) << "single-camera per-frame latency:\n"
              << "  iters: " << iters << "\n"
              << "  p50:   " << percentile(ms, 0.5) << " ms\n"
              << "  p95:   " << percentile(ms, 0.95) << " ms\n"
              << "  p99:   " << percentile(ms, 0.99) << " ms\n";
    return EXIT_SUCCESS;
}
