# Tracker backend comparison

The repository ships **two** single-camera tracker backends that share
the `ITracker` interface. This page summarises the trade-offs, and
explains why a third one people ask about is not here.

## Quick reference

|                       | IoU         | BYTETrack            |
|-----------------------|-------------|----------------------|
| Algorithm complexity  | minimal     | moderate             |
| CPU cost / frame      | <0.1 ms     | ~0.5 ms              |
| Motion model          | none        | Kalman 8-state       |
| Handles occlusion     | poor        | good (low-conf pass) |
| ID switch frequency   | high        | low                  |
| Dependency footprint  | header-only | header-only          |
| When to use it        | baseline    | default              |

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

## NvDCF - why it is not here

NvDCF is not a tracker anyone reimplements: it lives inside
DeepStream's `libnvds_nvmultiobjecttracker.so`. A project like this
one cannot run it in-process. The only way to use it is to build a
DeepStream pipeline and let a src-pad probe forward the tracker
metadata that plugin already produced.

That pipeline is on the roadmap and is not in this tree, so there is
no NvDCF backend to select. `tracker.type: nvdcf` parses - the schema
keeps it so an existing config gets a useful message rather than
"unknown tracker" - and then `make_tracker` throws:

> the NvDCF tracker requires a DeepStream pipeline that is not
> implemented in this reference; use bytetrack or iou

An earlier revision of this file described a wrapper that ingested
probe output. The wrapper existed, but nothing ever called its
`ingest_tracker_results`, so selecting `nvdcf` produced a tracker that
returned an empty result for every frame. It was removed rather than
documented.

## Choosing the right backend

- **You are running a single camera and want a quick demo.** IoU is
  fine.
- **You are running a single camera in production and worry about
  ID consistency.** BYTETrack.
- **You already have a DeepStream pipeline and want NvDCF.** Keep the
  pipeline you have and take only the cross-camera half of this repo:
  `IdentityMatcher` consumes `Track` values and does not care which
  backend produced them. Feeding it from your own probe is a smaller
  job than the DeepStream path on this roadmap.

The cross-camera matcher is independent of which backend you pick;
both produce the same `Track` value type so the orchestrator treats
them interchangeably.
