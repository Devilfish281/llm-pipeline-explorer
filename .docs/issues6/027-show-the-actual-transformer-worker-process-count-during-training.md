---
workflow: engineering-prompt-chain
document_type: implementation_ticket
prompt_name: to-tickets-prompt
status: ready-for-agent
triage_label: ready-for-agent
ticket_number: "027"
source_document: SPEC.md
recommended_next_prompt: implement-prompt
---

# Ticket 027: Show the Actual Transformer Worker-Process Count During Training

## What to build

Expose the actual number of spawned Transformer worker processes to learners as presentation text in the first public training sample. The displayed count must come from the existing actual worker-count boundary used by the current Transformer Training Run, be shown exactly once, and remain completely outside the Generated Text Sample, numerical training state, deterministic random streams, completion payload, and persisted Saved Transformer Model.

The training workflow must otherwise remain unchanged and authoritative: each request still starts from fresh weights, creates its Request-Scoped Worker Group, streams the existing events, and persists the complete model before successful completion.

## Acceptance Criteria

- [ ] The first public `epoch` event sample begins exactly with `Transformer worker processes: <actualWorkerCount>`, followed by one blank line and the unchanged Generated Text Sample text.
- [ ] The displayed count is obtained from the existing actual worker count calculated once for that run from one `os.cpu_count()` observation and bounded to one through four.
- [ ] Worker-count boundary cases preserve the current process rule: unavailable or nonpositive observations produce `1`, values `1`, `2`, and `4` produce those counts, and values above four produce `4`.
- [ ] The worker-process label appears only in the first public training epoch sample and is absent from every later epoch sample.
- [ ] Raw Generated Text Sample records contain only generated text and never contain the worker-process label.
- [ ] The training `done.samples` collection remains unchanged and contains no worker-process label.
- [ ] Existing deterministic Generated Text Sample fixtures and generated text remain unchanged when compared before presentation formatting.
- [ ] Tokenization, Transformer inputs, random streams, loss values, parameter updates, final parameters, persistence metadata, and Saved Transformer Model contents are unchanged by the label.
- [ ] The training `epoch` payload retains exactly `epoch`, `loss`, and `sample`; no new public payload field is added for worker count.
- [ ] The training `done` payload retains its approved architecture, final-loss, and sample structure.
- [ ] Saved Transformer Generation Runs display no worker-process label and never synthesize `Transformer worker processes: 0` or any other count.
- [ ] The wording uses exactly `Transformer worker processes` and does not claim physical cores, affinity, or guaranteed hardware parallelism.
- [ ] Existing fresh-weight initialization, Request-Scoped Worker Group creation, four Logical Training Shards, event order, cleanup, and persistence-before-`done` regressions remain preserved.

## Testing Expectations

- **Approved test seam:** The affected `POST /train-transformer` HTTP/SSE behavior through FastAPI `TestClient`, combined with the existing stable worker-count and raw Generated Text Sample public boundaries.
- **Behavior to verify:** Dynamic bounded count, exact first-sample prefix and blank line, single occurrence, unchanged later samples and payload keys, absence from raw samples and persistence, unchanged deterministic fixtures, and absence from saved-model generation.
- **Relevant prior art:** Existing actual-worker-count tests, Transformer route exact SSE tests, Generated Text Sample deterministic fixtures, completion payload tests, worker-group tests, and persistence tests.
- **Do not test through:** Private formatting helper names, exact local string-concatenation steps, operating-system process affinity, physical CPU topology, or internal route variable names.

## Blocked By

- None — can start immediately.

## Constraints and Out of Scope

- This is presentation-only; do not add worker count to model input, tokenization, numerical state, result records, persistence, or SSE payload fields.
- Do not change worker-count calculation, worker assignment, four Logical Training Shards, worker protocol, shared memory, optimizer, or training mathematics.
- Do not show the label during `POST /load-transformer`.
- Do not redesign the frontend or add a new control.
- Keep ADR 0002 as the authority for Transformer training and process lifecycle.

## Source

- `SPEC.md` — training preservation, worker-count source, exact label format, and presentation boundary.
- `GRILL_WITH_DOCS_RESULT.md` — confirmed worker-process visibility behavior.
- `CONTEXT.md` — canonical Transformer Training Run and Request-Scoped Worker Group terminology.
- ADR 0002 — existing training and worker lifecycle authority.
- ADR 0003 — confirms the label is training-only and saved-model generation creates no worker group.
- Latest Python Backend source export supplied with the source specification.

## Recommended Next Step

Run `implement-prompt` in a fresh conversation using this ticket, the source specification, and relevant project files.
