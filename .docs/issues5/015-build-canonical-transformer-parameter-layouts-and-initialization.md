---
workflow: engineering-prompt-chain
document_type: implementation_ticket
prompt_name: to-tickets-prompt
status: ready-for-agent
triage_label: ready-for-agent
ticket_number: "015"
source_document: SPEC.md
recommended_next_prompt: implement-prompt
---

# Ticket 015: Build canonical Transformer parameter layouts and initialization

## What to build

Provide the sole canonical authority for Transformer parameter names, layer ownership, flat float offsets, lengths, shapes, total parameter counts, NumPy views, and fresh Weight Initialization for every supported model depth.

The layout must be stable across parent code, spawned workers, shared-memory attachment, optimizer state, model conversion, and tests. Initialization must consume the shared Mulberry32 stream in the exact approved traversal order and materialize every completed coordinate immediately as `float32`.

## Acceptance Criteria

- [ ] Layouts are supported for every requested layer count from one through six while all other architecture values remain fixed at context length `32`, embedding dimension `32`, two heads, head dimension `16`, and feed-forward dimension `128`.
- [ ] The flat parameter order is exactly `tokEmb`, `posEmb`, every block's approved sixteen arrays in layer order, `lnFGamma`, `lnFBeta`, `headW`, and `headB`.
- [ ] Every layout record exposes stable key, block identity where applicable, float offset, length, shape, and total parameter count information needed by parent and worker boundaries.
- [ ] All parameter regions are contiguous, non-overlapping, gap-free, and end exactly at the reported total float count.
- [ ] Independent fixtures assert exact offsets, shapes, lengths, and total parameter counts for representative one-, two-, and six-layer configurations and complete structural invariants for all depths.
- [ ] A view builder maps one sufficiently sized flat buffer into exact C-order NumPy `float32` views using byte offset equal to float offset multiplied by four.
- [ ] View creation rejects undersized storage, incorrect dtype, non-contiguous backing storage, inconsistent layouts, and any region outside the canonical range.
- [ ] Capacity greater than the logical required bytes is accepted while every view remains limited to the exact canonical range.
- [ ] Weight Initialization uses one request-owned shared Mulberry32 generator and consumes draws in the approved order: each block's `wQ`, `wK`, `wV`, `wO`, `ff1W`, and `ff2W`, followed by `tokEmb`, `posEmb`, and `headW`.
- [ ] Every Xavier coordinate uses `sqrt(6 / (fan_in + fan_out))`, consumes one draw, and is stored immediately as `float32` before the next coordinate is calculated.
- [ ] Bias arrays and Layer Normalization beta arrays are exactly zero, gamma arrays are exactly one, and deterministic fills consume no random draws.
- [ ] Selected coordinates, complete draw counts, final generator state, deterministic fills, and flat-array checksums match independent reference fixtures.
- [ ] Two runs with identical configuration and seed create equivalent but non-aliased weight storage, while different layer counts use their own exact layouts.
- [ ] All initialized weights are finite before they can be used by forward computation or shared memory.
- [ ] No existing Saved Transformer Model is read or used during initialization.

## Testing Expectations

- **Approved test seam:** The stable public Transformer layout, view-construction, parameter-count, and Weight Initialization boundary.
- **Behavior to verify:** Exact flat ordering, offsets, shapes, parameter counts, buffer validation, C-order float32 views, Mulberry32 draw order, Xavier values, deterministic fills, finiteness, and run isolation.
- **Relevant prior art:** The existing exact fixture style for deterministic initial state plus the accepted canonical-layout contract.
- **Do not test through:** Private record class names, a particular dictionary implementation, local loop variables, shared-memory names, or HTTP configuration parsing.

## Blocked By

- [Ticket 012 — Centralize TypeScript-compatible randomness and rounding](012-centralize-typescript-compatible-randomness-and-rounding.md)

## Constraints and Out of Scope

- One canonical layout builder must serve parent, worker, optimizer, persistence, and test consumers; do not maintain parallel offset tables.
- Do not change architecture dimensions other than the approved requested layer count.
- Do not use NumPy-native randomness or vectorized random fills that change draw order or float32 materialization order.
- Do not load, resume, cache, or skip initialization from a saved model.
- Do not implement full forward/backward mathematics, optimizer updates, workers, routing, or persistence in this ticket.
- No new dependency or lockfile change is expected.

## Source

- `SPEC.md` — fixed architecture, canonical layout, initialization order, and exact fixture requirements.
- `CONTEXT.md` — Weight Initialization and Transformer Training Compatibility terminology.
- ADR 0002 — canonical flat order, float32 views, and TypeScript-compatible Xavier traversal.
- Ticket 012 — shared deterministic utility boundary.
- Latest TypeScript Reference Implementation — weight structures and initialization evidence.

## Recommended Next Step

Run `implement-prompt` in a fresh conversation using this ticket, its blocker tickets, `SPEC.md`, `CONTEXT.md`, ADR 0002, the latest Python Backend source export, and the latest TypeScript Reference Implementation.
