# Cross-camera matching

The matcher's job is to assign a stable `global_id` to every
physical person, even when they cross between cameras and the
per-camera tracker has issued them a fresh `local_id`.

## Three gates

A candidate pairing (new observation X, existing global Y) is
*infeasible* unless **all three** of these hold:

1. **Zone transition is legal.** Configured in `cameras.yaml`. An
   empty `transitions:` block means any-to-any.
2. **Time delta is reasonable.** `spatial_overlap_window_ms` (5 s
   by default) is the upper bound. Larger windows admit more
   matches but also more false positives across long pauses.
3. **ReID similarity is above the threshold.** `reid_threshold`
   (0.7 by default) is a cosine similarity floor.

Pairings that pass all three are scored as `1 - cosine_similarity`
and fed into `solve_assignment`. The optimal pairing is then
filtered against `hungarian_cost_cap` so an assignment that just
barely passed the threshold is still rejected when it is the only
plausible candidate.

## Why a cap on top of a threshold?

Without the cap, a track that *could* match any of three global ids
- with all three having identical similarity - would still get a
single arbitrary assignment. The cap rejects ambiguous-looking
assignments and lets the matcher fall back to spawning a new
global_id, which the next frame will sort out cleanly.

## Embedding source: latest vs gallery

By default the matcher scores against the most recent embedding on
the new observation. When `register_gallery()` has been called for
a camera, the matcher *also* queries that camera's `ReidGallery`
for the best similarity over its rolling bank, taking the max of
the two. This is the single biggest robustness win - a person who
turned their head between cameras is otherwise scored against a
single non-canonical view.

## Spatial-temporal failure modes

- **Clock skew between cameras.** If camera A's wall clock is 800
  ms ahead of camera B's, a transition that took 200 ms will look
  like a *backwards* time travel. The matcher rejects negative
  deltas; sync your cameras with PTP / NTP and verify with a
  stopwatch.
- **A person standing still.** Stationary people produce many
  observations on the same camera, no transition. The matcher
  carries forward the existing `(camera, local_id) -> global` map
  so this is fine - until the per-camera tracker drops the track
  due to occlusion. When the new local_id appears, it has to look
  up via ReID against an old canonical embedding. This is exactly
  what the gallery is for.
- **Two physically distinct people who look very similar.**
  Identical twins, or the same uniform on two people. The matcher
  does not have a unique signal to distinguish these and will
  conflate them. Use the zone-transition table to disambiguate
  when possible (twin A is in the lobby, twin B in the corridor).

## Gating by spatial geometry

The current matcher does not consider camera homographies; it only
uses the zone-transition table as a coarse spatial constraint.
A future revision will accept a per-camera ground-plane homography
so the matcher can reject transitions that are geometrically
impossible (the person would have had to walk through a wall).
