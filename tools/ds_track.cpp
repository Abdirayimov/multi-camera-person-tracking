// mc_tracking_ds — the DeepStream + NvDCF driver.
//
// Runs every camera in cameras.yaml through
//   nvurisrcbin -> nvstreammux -> nvinfer (YOLO person) -> nvtracker (NvDCF)
// and logs the per-camera tracks the src-pad probe hands back. This is the
// path where NvDCF actually runs; the OpenCV driver (mc_tracking_video)
// rejects tracker.type nvdcf for exactly this reason.

#include <csignal>
#include <cstdint>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <string>

#include "mc_tracking/config/system_config.hpp"
#include "mc_tracking/pipeline/deepstream_pipeline.hpp"
#include "mc_tracking/utils/logger.hpp"

namespace {

mc_tracking::pipeline::DeepStreamPipeline* g_pipeline = nullptr;

void handle_signal(int) {
    if (g_pipeline != nullptr) {
        g_pipeline->stop();
    }
}

void usage() {
    std::cout << "mc_tracking_ds --config <system_config.yaml> --cameras <cameras.yaml>\n"
              << "               [--pgie <pgie_yolo_person.txt>]\n\n"
              << "Runs the DeepStream + NvDCF pipeline over every camera in\n"
              << "cameras.yaml and logs per-camera track batches. The NvDCF\n"
              << "element is configured from tracker.nvdcf in the system config.\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string config_path;
    std::string cameras_path;
    std::string pgie_path = "configs/pgie_yolo_person.txt";

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--cameras" && i + 1 < argc) {
            cameras_path = argv[++i];
        } else if (arg == "--pgie" && i + 1 < argc) {
            pgie_path = argv[++i];
        } else {
            usage();
            return arg == "--help" ? 0 : 1;
        }
    }
    if (config_path.empty() || cameras_path.empty()) {
        usage();
        return 1;
    }

    try {
        const auto cfg = mc_tracking::config::SystemConfig::load(config_path);
        const auto cams = mc_tracking::config::CamerasConfig::load(cameras_path);
        mc_tracking::utils::init_logger(cfg.logging.level, cfg.logging.json);

        mc_tracking::pipeline::DeepStreamPipeline pipeline(cfg, pgie_path);

        // Per-camera tallies, filled from the probe's streaming thread.
        std::mutex tally_mutex;
        std::map<std::string, std::set<std::uint64_t>> ids_seen;
        std::map<std::string, std::uint64_t> frames_seen;

        pipeline.set_track_callback([&](const mc_tracking::pipeline::CameraFrameResult& frame) {
            std::lock_guard<std::mutex> lock(tally_mutex);
            auto& ids = ids_seen[frame.camera_id];
            for (const auto& t : frame.tracks) {
                ids.insert(t.local_id);
            }
            auto& n = frames_seen[frame.camera_id];
            if (++n % 30 == 1) {
                MCT_LOG_INFO("[{}] frame {}: {} active tracks, {} unique ids so far",
                             frame.camera_id, frame.frame_number, frame.tracks.size(), ids.size());
            }
        });

        for (const auto& cam : cams.cameras) {
            if (!pipeline.add_source(cam.id, cam.uri)) {
                MCT_LOG_ERROR("could not add source '{}', aborting", cam.id);
                return 1;
            }
        }

        g_pipeline = &pipeline;
        std::signal(SIGINT, handle_signal);
        std::signal(SIGTERM, handle_signal);

        pipeline.start();
        pipeline.wait();
        g_pipeline = nullptr;

        std::lock_guard<std::mutex> lock(tally_mutex);
        for (const auto& [cam, ids] : ids_seen) {
            MCT_LOG_INFO("[{}] done: {} frames, {} unique NvDCF track ids", cam, frames_seen[cam],
                         ids.size());
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "fatal: " << e.what() << "\n";
        return 1;
    }
}
