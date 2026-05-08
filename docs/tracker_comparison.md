# Tracker backend comparison

The repository ships three single-camera tracker backends that share
the `ITracker` interface. This page summarises the trade-offs.

## Quick reference

|                       | IoU         | BYTETrack            | NvDCF (DS)         |
|-----------------------|-------------|----------------------|--------------------|
| Algorithm complexity  | minimal     | moderate             | proprietary        |
| CPU cost / frame      | <0.1 ms     | ~0.5 ms              | n/a (GStreamer)    |
| Motion model          | none        | Kalman 8-state       | Kalman + visual    |
| Handles occlusion     | poor        | good (low-conf pass) | excellent          |
| ID switch frequency   | high        | low                  | very low           |
| Dependency footprint  | header-only | header-only          | DeepStream SDK     |
| When to use it        | baseline    | default              | full DS pipeline   |

## IoU tracker

Greedy IoU matching with a fixed threshold. No motion prediction:
boxes that move significantly between frames break association.
Useful as a comparison baseline; do not deploy on its own.

## BYTETrack

The default, and what the repo's overlay screenshots are produced
with. The trick BYTETrack adds over plain SORT is the *second-stage*
association pass against low-confidence detections - those are
exactly the ones produced under partial occlusion, and recovering
their IDs collapses the ID-switch rate substantially.

Tuning notes:

- `high_thresh=0.5` is a good detector-agnostic default. If you find
  the tracker dropping confirmed tracks too easily, drop to 0.4.
- `low_thresh=0.1` is intentionally permissive. The second-stage
  pass is the safety net; lowering this further rarely helps and
  starts admitting noise.
- `track_buffer=30` (frames) governs how long a lost track waits
  before eviction. For 30 fps video that is one second; raise it
  if your scenes have longer occlusions.
- `match_thresh=0.8` is the (1 - IoU) cost cap. Tightening to 0.7
  reduces ID switches at the cost of more new-track spawns.

## NvDCF wrapper

The wrapper here does not run NvDCF in-process; it ingests results
that the DeepStream pipeline's src-pad probe forwards from
`libnvds_nvmultiobjecttracker.so`. The benefit is that you keep the
mature DeepStream tracker - including its visual feature extractor -
without abandoning the rest of this toolkit.

Build with `BUILD_DEEPSTREAM=ON`. When DeepStream is missing, the
NvDCF wrapper is excluded from the build automatically and choosing
`tracker.type: nvdcf` at runtime throws a clear error.

## Choosing the right backend

- **You are running a single camera and want a quick demo.** IoU is
  fine.
- **You are running a single camera in production and worry about
  ID consistency.** BYTETrack.
- **You already have a DeepStream pipeline and want the best
  available tracker.** NvDCF.

The cross-camera matcher is independent of which backend you pick;
all three produce the same `Track` value type so the orchestrator
treats them interchangeably.
