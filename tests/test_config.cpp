#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

#include "mc_tracking/config/system_config.hpp"

namespace {

using mc_tracking::config::CameraEntry;
using mc_tracking::config::CamerasConfig;
using mc_tracking::config::SystemConfig;
using mc_tracking::config::TrackerType;
using mc_tracking::config::ZoneTransition;

std::string fixture(const std::string& name) {
    return std::string(MCT_TEST_FIXTURES) + "/" + name;
}

std::string repo_config(const std::string& name) {
    return std::string(MCT_TEST_FIXTURES) + "/../../configs/" + name;
}

/// A config with every field valid, used as the starting point for the
/// programmatic `validate()` tests.
SystemConfig valid_config() {
    SystemConfig cfg;
    cfg.detection.engine_path = "models/engines/detector.engine";
    cfg.reid.enabled = false;
    return cfg;
}

// ------------------------------------------------------------- happy path

TEST(SystemConfigLoad, ReadsEveryFieldFromAFullyPopulatedFile) {
    const SystemConfig cfg = SystemConfig::load(fixture("valid_full.yaml"));

    EXPECT_EQ(cfg.pipeline.muxer_width, 1920u);
    EXPECT_EQ(cfg.pipeline.muxer_height, 1080u);
    EXPECT_EQ(cfg.pipeline.batch_size, 4u);
    EXPECT_FALSE(cfg.pipeline.emit_overlay);

    EXPECT_EQ(cfg.detection.engine_path, "models/engines/yolo11x_person_fp16.engine");
    EXPECT_EQ(cfg.detection.input_width, 1280u);
    EXPECT_EQ(cfg.detection.input_height, 1280u);
    EXPECT_FLOAT_EQ(cfg.detection.confidence_threshold, 0.35f);
    EXPECT_FLOAT_EQ(cfg.detection.nms_iou_threshold, 0.6f);
    EXPECT_EQ(cfg.detection.person_class_id, 0);

    EXPECT_EQ(cfg.tracker.type, TrackerType::Iou);
    EXPECT_FLOAT_EQ(cfg.tracker.bytetrack.high_thresh, 0.55f);
    EXPECT_FLOAT_EQ(cfg.tracker.bytetrack.low_thresh, 0.15f);
    EXPECT_FLOAT_EQ(cfg.tracker.bytetrack.new_track_thresh, 0.65f);
    EXPECT_EQ(cfg.tracker.bytetrack.track_buffer, 45u);
    EXPECT_FLOAT_EQ(cfg.tracker.bytetrack.match_thresh, 0.75f);
    EXPECT_FLOAT_EQ(cfg.tracker.bytetrack.aspect_ratio_thresh, 1.8f);
    EXPECT_FLOAT_EQ(cfg.tracker.iou.iou_thresh, 0.4f);
    EXPECT_EQ(cfg.tracker.iou.max_age, 20u);
    EXPECT_EQ(cfg.tracker.iou.min_hits, 2u);
    EXPECT_EQ(cfg.tracker.nvdcf.tracker_width, 960u);
    EXPECT_EQ(cfg.tracker.nvdcf.tracker_height, 544u);

    EXPECT_TRUE(cfg.reid.enabled);
    EXPECT_EQ(cfg.reid.embedding_dim, 512u);
    EXPECT_EQ(cfg.reid.gallery_size_per_track, 12u);

    EXPECT_FALSE(cfg.crosscam.enabled);
    EXPECT_FLOAT_EQ(cfg.crosscam.reid_threshold, 0.8f);
    EXPECT_EQ(cfg.crosscam.spatial_overlap_window_ms, 2500u);
    EXPECT_FLOAT_EQ(cfg.crosscam.hungarian_cost_cap, 0.25f);

    EXPECT_EQ(cfg.logging.level, "debug");
    EXPECT_FALSE(cfg.logging.json);
}

TEST(SystemConfigLoad, FillsInDocumentedDefaultsForOmittedFields) {
    const SystemConfig cfg = SystemConfig::load(fixture("minimal.yaml"));

    EXPECT_EQ(cfg.pipeline.muxer_width, 1280u);
    EXPECT_EQ(cfg.pipeline.muxer_height, 720u);
    EXPECT_EQ(cfg.pipeline.batch_size, 1u);
    EXPECT_TRUE(cfg.pipeline.emit_overlay);

    EXPECT_EQ(cfg.detection.input_width, 640u);
    EXPECT_EQ(cfg.detection.input_height, 640u);
    EXPECT_FLOAT_EQ(cfg.detection.confidence_threshold, 0.4f);
    EXPECT_FLOAT_EQ(cfg.detection.nms_iou_threshold, 0.5f);

    // An absent `tracker` section leaves the struct defaults in place.
    EXPECT_EQ(cfg.tracker.type, TrackerType::ByteTrack);
    EXPECT_FLOAT_EQ(cfg.tracker.bytetrack.high_thresh, 0.5f);
    EXPECT_EQ(cfg.tracker.bytetrack.track_buffer, 30u);
    EXPECT_FLOAT_EQ(cfg.tracker.iou.iou_thresh, 0.3f);
    EXPECT_EQ(cfg.tracker.iou.min_hits, 3u);

    EXPECT_FLOAT_EQ(cfg.crosscam.reid_threshold, 0.7f);
    EXPECT_EQ(cfg.crosscam.spatial_overlap_window_ms, 5000u);
    EXPECT_FLOAT_EQ(cfg.crosscam.hungarian_cost_cap, 0.4f);

    EXPECT_EQ(cfg.logging.level, "info");
    EXPECT_TRUE(cfg.logging.json);
}

TEST(SystemConfigLoad, AcceptsEveryConfigShippedWithTheRepository) {
    for (const char* name : {"system_config.yaml", "demo_pedestrian.yaml", "demo_crosscam.yaml"}) {
        EXPECT_NO_THROW({
            const SystemConfig cfg = SystemConfig::load(repo_config(name));
            (void)cfg;
        }) << name;
    }
}

// ---------------------------------------------------------------- failures

TEST(SystemConfigLoad, ReportsAMissingFileByName) {
    const std::string path = fixture("this_file_does_not_exist.yaml");

    try {
        SystemConfig::load(path);
        FAIL() << "expected a std::runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find(path), std::string::npos)
            << "error should name the offending path: " << e.what();
    }
}

TEST(SystemConfigLoad, RejectsMalformedYaml) {
    EXPECT_THROW(SystemConfig::load(fixture("malformed.yaml")), std::runtime_error);
}

TEST(SystemConfigLoad, RejectsAMissingDetectionSection) {
    EXPECT_THROW(SystemConfig::load(fixture("missing_detection.yaml")), std::runtime_error);
}

TEST(SystemConfigLoad, RejectsAMissingRequiredKey) {
    try {
        SystemConfig::load(fixture("missing_engine_path.yaml"));
        FAIL() << "expected a std::runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("engine_path"), std::string::npos)
            << "error should name the missing key: " << e.what();
    }
}

TEST(SystemConfigLoad, RejectsAConfidenceThresholdAboveOne) {
    try {
        SystemConfig::load(fixture("invalid_confidence.yaml"));
        FAIL() << "expected a std::runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("detection.confidence_threshold"), std::string::npos)
            << "error should name the offending key: " << e.what();
    }
}

TEST(SystemConfigLoad, RejectsInvertedByteTrackThresholds) {
    try {
        SystemConfig::load(fixture("inverted_bytetrack_thresholds.yaml"));
        FAIL() << "expected a std::runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("low_thresh"), std::string::npos)
            << "error should explain which pair is inconsistent: " << e.what();
    }
}

TEST(SystemConfigLoad, RejectsEnabledReidWithNoEngine) {
    EXPECT_THROW(SystemConfig::load(fixture("reid_without_engine.yaml")), std::runtime_error);
}

TEST(SystemConfigLoad, RejectsAnUnknownTrackerType) {
    try {
        SystemConfig::load(fixture("unknown_tracker.yaml"));
        FAIL() << "expected a std::runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("deepsort"), std::string::npos)
            << "error should echo the unknown value: " << e.what();
    }
}

// ---------------------------------------------------- programmatic checks

TEST(SystemConfigValidate, AcceptsTheStructDefaultsPlusAnEnginePath) {
    EXPECT_NO_THROW(valid_config().validate());
}

TEST(SystemConfigValidate, RejectsAnEmptyDetectorEnginePath) {
    SystemConfig cfg = valid_config();
    cfg.detection.engine_path.clear();
    EXPECT_THROW(cfg.validate(), std::runtime_error);
}

TEST(SystemConfigValidate, RejectsAZeroBatchSize) {
    SystemConfig cfg = valid_config();
    cfg.pipeline.batch_size = 0;
    EXPECT_THROW(cfg.validate(), std::runtime_error);
}

TEST(SystemConfigValidate, RejectsANegativePersonClassId) {
    SystemConfig cfg = valid_config();
    cfg.detection.person_class_id = -1;
    EXPECT_THROW(cfg.validate(), std::runtime_error);
}

TEST(SystemConfigValidate, RejectsAZeroTrackBuffer) {
    // A zero buffer evicts a track the instant it is occluded, which
    // defeats the point of the lost pool.
    SystemConfig cfg = valid_config();
    cfg.tracker.bytetrack.track_buffer = 0;
    EXPECT_THROW(cfg.validate(), std::runtime_error);
}

TEST(SystemConfigValidate, RejectsAZeroMinHits) {
    SystemConfig cfg = valid_config();
    cfg.tracker.iou.min_hits = 0;
    EXPECT_THROW(cfg.validate(), std::runtime_error);
}

TEST(SystemConfigValidate, AcceptsEqualByteTrackThresholds) {
    // Equal is degenerate but legal: it just leaves the low band empty.
    SystemConfig cfg = valid_config();
    cfg.tracker.bytetrack.high_thresh = 0.5f;
    cfg.tracker.bytetrack.low_thresh = 0.5f;
    EXPECT_NO_THROW(cfg.validate());
}

TEST(SystemConfigValidate, AcceptsTheReidThresholdBoundaries) {
    SystemConfig cfg = valid_config();
    cfg.crosscam.reid_threshold = 1.0f;
    EXPECT_NO_THROW(cfg.validate());
    cfg.crosscam.reid_threshold = -1.0f;
    EXPECT_NO_THROW(cfg.validate());
}

TEST(SystemConfigValidate, RejectsAReidThresholdOutsideTheCosineRange) {
    SystemConfig cfg = valid_config();
    cfg.crosscam.reid_threshold = 1.5f;
    EXPECT_THROW(cfg.validate(), std::runtime_error);
}

TEST(SystemConfigValidate, IgnoresReidSettingsWhileReidIsOff) {
    SystemConfig cfg = valid_config();
    cfg.reid.enabled = false;
    cfg.reid.engine_path.clear();
    cfg.reid.embedding_dim = 0;
    EXPECT_NO_THROW(cfg.validate());
}

// ------------------------------------------------------------- cameras

TEST(CamerasConfigLoad, ReadsCamerasAndTransitions) {
    const CamerasConfig cfg = CamerasConfig::load(fixture("cameras_valid.yaml"));

    ASSERT_EQ(cfg.cameras.size(), 2u);
    EXPECT_EQ(cfg.cameras[0].id, "cam_hall");
    EXPECT_EQ(cfg.cameras[0].zone, "hall");
    EXPECT_EQ(cfg.cameras[1].id, "cam_lobby");
    ASSERT_EQ(cfg.transitions.size(), 2u);
    EXPECT_FALSE(cfg.any_transition_allowed());
}

TEST(CamerasConfigLoad, AcceptsTheCameraFilesShippedWithTheRepository) {
    for (const char* name : {"cameras.yaml", "demo_crosscam_cameras.yaml"}) {
        EXPECT_NO_THROW({
            const CamerasConfig cfg = CamerasConfig::load(repo_config(name));
            (void)cfg;
        }) << name;
    }
}

TEST(CamerasConfigLoad, ReportsAMissingFileByName) {
    const std::string path = fixture("no_such_cameras.yaml");

    try {
        CamerasConfig::load(path);
        FAIL() << "expected a std::runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find(path), std::string::npos) << e.what();
    }
}

TEST(CamerasConfigLoad, RejectsDuplicateCameraIds) {
    try {
        CamerasConfig::load(fixture("cameras_duplicate_id.yaml"));
        FAIL() << "expected a std::runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("cam_hall"), std::string::npos) << e.what();
    }
}

TEST(CamerasConfigLoad, RejectsATransitionNamingAnUndeclaredZone) {
    try {
        CamerasConfig::load(fixture("cameras_unknown_zone.yaml"));
        FAIL() << "expected a std::runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("roof"), std::string::npos) << e.what();
    }
}

TEST(CamerasConfig, TreatsAnEmptyTransitionTableAsAnyToAny) {
    CamerasConfig cfg;
    cfg.cameras.push_back(CameraEntry{"a", "file://a.mp4", "hall"});

    EXPECT_TRUE(cfg.any_transition_allowed());
    EXPECT_TRUE(cfg.transition_allowed("hall", "roof"));
}

TEST(CamerasConfig, HonoursADeclaredTransitionInOneDirectionOnly) {
    CamerasConfig cfg;
    cfg.cameras.push_back(CameraEntry{"a", "file://a.mp4", "hall"});
    cfg.cameras.push_back(CameraEntry{"b", "file://b.mp4", "lobby"});
    cfg.transitions.push_back(ZoneTransition{"hall", "lobby"});

    EXPECT_TRUE(cfg.transition_allowed("hall", "lobby"));
    EXPECT_FALSE(cfg.transition_allowed("lobby", "hall"))
        << "the reverse direction has to be declared explicitly";
}

TEST(CamerasConfigValidate, RejectsACameraWithNoUri) {
    CamerasConfig cfg;
    cfg.cameras.push_back(CameraEntry{"a", "", "hall"});
    EXPECT_THROW(cfg.validate(), std::runtime_error);
}

TEST(CamerasConfigValidate, RejectsACameraWithNoId) {
    CamerasConfig cfg;
    cfg.cameras.push_back(CameraEntry{"", "file://a.mp4", "hall"});
    EXPECT_THROW(cfg.validate(), std::runtime_error);
}

TEST(CamerasConfigValidate, AcceptsCamerasWithNoZoneDeclared) {
    CamerasConfig cfg;
    cfg.cameras.push_back(CameraEntry{"a", "file://a.mp4", ""});
    EXPECT_NO_THROW(cfg.validate());
}

}  // namespace
