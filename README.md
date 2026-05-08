<h1 align="center">multi-camera-person-tracking</h1>

<p align="center">
  <i>C++ pipeline for tracking people across many cameras with YOLOv8 + a pluggable single-camera tracker (BYTETrack / IoU / NvDCF), OSNet ReID, and Hungarian-assigned cross-camera identity matching.</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/C%2B%2B-17-blue.svg" alt="C++17">
  <img src="https://img.shields.io/badge/CUDA-12.x-76B900.svg" alt="CUDA">
  <img src="https://img.shields.io/badge/TensorRT-8.6%2B-76B900.svg" alt="TensorRT">
  <img src="https://img.shields.io/badge/DeepStream-7.x%20|%208.x-76B900.svg" alt="DeepStream">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey.svg" alt="License">
  <img src="https://img.shields.io/badge/status-reference%20implementation-orange.svg" alt="Status">
</p>

---

## Why this exists

Single-camera tracking is largely solved. Cross-camera tracking is
where the engineering gets interesting:

- **Identity hand-off.** A person leaves camera A and enters
  camera B six seconds later. The tracker on B will assign a fresh
  local id; you need a separate matcher to recognise that the
  observation belongs to the same physical person.
- **Constraints to keep matches sane.** "Looks similar" is not
  enough on its own. A robust matcher gates ReID similarity by
  zone-transition rules (was this even a plausible move?) and by a
  spatial-temporal window (the same body cannot be in two
  non-overlapping cameras at the same instant).
- **Rolling appearance bank per track.** Collapsing a track to one
  embedding throws away viewpoint variance; matching against the
  *best* of a small per-track gallery preserves both face-on and
  facing-away signatures.
- **Pluggable tracker backends.** Different sites need different
  trade-offs - IoU is cheapest, BYTETrack handles partial
  occlusion, NvDCF integrates with a DeepStream pipeline.
  A clean tracker interface lets you A/B them without rewriting
  the rest of the pipeline.

The repository is a clean-room reference implementation of those
patterns. All algorithms are public; the code is original.

## What's inside

- **YOLOv8 person detector** with letterboxed 640x640 input
- **Three single-camera tracker backends behind a common interface:**
    - `ByteTrack` - two-stage cascade (Zhang et al., ECCV 2022)
    - `IouTracker` - greedy IoU baseline
    - `NvDcfTracker` - thin wrapper over DeepStream's NvDCF
- **OSNet ReID** (Zhou et al., ICCV 2019) embedding extractor
- **Per-track rolling appearance gallery** (`ReidGallery`)
- **Cross-camera matcher** with zone-transition + spatial-temporal
  gating and Hungarian assignment
- **OpenCV-based offline driver** that runs single-cam or multi-cam
- **Docker + docker-compose** (DeepStream-devel base)
- **Benchmark CLI** for per-frame latency

## Architecture

```
                       per camera
   ┌────────────────────────────────────────────────────────┐
   │  frame  →  YOLOv8  →  tracker (bytetrack/iou/nvdcf)   │
   │             ↓            ↓                             │
   │         person crops    confirmed tracks (local_id)    │
   │             ↓                                          │
   │         OSNet ReID                                     │
   │             ↓                                          │
   │      per-track gallery                                 │
   └─────────────────┬──────────────────────────────────────┘
                     │ CameraFrameResult
                     ▼
          MultiCameraOrchestrator
                     │
                     ▼
          IdentityMatcher (Hungarian over ReID cosine,
                          gated by zone topology
                          + spatial-overlap window)
                     │
                     ▼
       global_id stamped back on every Track
                     │
                     ▼
                 Visualizer  →  annotated mp4 per camera
```

## Quick start

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Acquire ONNX checkpoints (script prints instructions):
./scripts/download_models.sh

# Compile FP16 engines:
./scripts/build_engines.sh

# Single camera:
./scripts/infer_video.sh input.mp4 annotated.mp4

# Multi camera:
./build/mc_tracking_video \
    --config configs/system_config.yaml \
    --cameras configs/cameras.yaml \
    --output-dir data/out
```

## Configuring the tracker

`configs/system_config.yaml` is the single source of truth:

```yaml
tracker:
  type: bytetrack         # or "iou", or "nvdcf"
  bytetrack:
    high_thresh: 0.5
    low_thresh: 0.1
    track_buffer: 30
    match_thresh: 0.8
  iou:
    iou_thresh: 0.3
    max_age: 30
    min_hits: 3
```

Switching backends is config-only; no rebuild required.

## Cross-camera matcher

`configs/cameras.yaml` declares each camera's `id`, `uri`, and
`zone`, plus the allowed zone-to-zone transitions. The matcher uses
those to throw out impossible candidate pairings before scoring:

```yaml
cameras:
  - { id: cam-01, uri: "...", zone: lobby }
  - { id: cam-02, uri: "...", zone: corridor }
transitions:
  - { from: lobby,    to: corridor }
  - { from: corridor, to: exit }
```

When `transitions` is empty, any-to-any pairing is allowed.

## Project structure

```
.
├── CMakeLists.txt
├── cmake/
├── configs/
│   ├── system_config.yaml          tracker / reid / crosscam
│   ├── cameras.yaml                multi-camera topology
│   ├── tracker_nvdcf.yml           DeepStream-native tracker config
│   └── pgie_yolov8_person.txt      gst-nvinfer config
├── docker/, docker-compose.yml
├── include/mc_tracking/
│   ├── config/, utils/
│   ├── tracker/   bytetrack, iou, nvdcf, kalman_filter
│   ├── reid/      osnet_extractor, reid_gallery
│   ├── crosscam/  identity_matcher, hungarian
│   ├── trt/       trt_engine, yolov8_detector
│   ├── pipeline/  single_camera, multi_camera
│   └── overlay/
├── src/                  (mirrors include/)
├── tools/                benchmark.cpp
├── scripts/              download_models, build_engines, infer_video
└── docs/
    ├── architecture.md
    ├── tracker_comparison.md
    └── crosscam_matching.md
```

## Performance

Indicative numbers on synthetic 720p input, RTX 3090, bytetrack +
OSNet enabled, single camera:

| Stage                        | p50 latency  |
|------------------------------|-------------:|
| YOLOv8s detect (1 frame)     | ~7 ms        |
| BYTETrack association        | <1 ms        |
| OSNet ReID (8 crops, batched)| ~3 ms        |

Cross-camera matching adds <0.5 ms/frame even with 64 active
global ids; the cost is dominated by the embedding extraction
upstream.

## Limitations

- The OpenCV driver runs cameras sequentially; for production use
  the DeepStream pipeline (BUILD_DEEPSTREAM=ON) and run cameras
  in parallel via separate `nvurisrcbin` sources.
- The Hungarian solver here is a simplified greedy-with-augment
  variant; for >=100 active global ids you may want to swap in a
  full Munkres implementation.
- ReID is camera-agnostic. A fully calibrated system would also
  use camera-to-camera homographies to constrain plausible
  reentry positions; this repo does not.
- INT8 quantisation of the OSNet engine is not validated end-to-end;
  FP16 is the documented configuration.

## Roadmap

- [ ] DeepStream pipeline that exposes the same probe-based
      orchestrator as the OpenCV path
- [ ] Camera homography support in the cross-camera matcher
- [ ] Optional embedding cache (Redis) for clusters spanning
      multiple processes
- [ ] Trajectory output (parquet) for downstream analytics

## License

MIT - see [LICENSE](LICENSE).

## About

Reference implementation of patterns from production multi-camera
tracking work. Algorithms are the published originals
(YOLOv8, BYTETrack, OSNet, Hungarian); the code is written from
scratch and uses public checkpoints + synthetic data.

Open to contract work on similar systems -
[email](mailto:khusanabdirayimov@gmail.com) -
[GitHub](https://github.com/Abdirayimov)
