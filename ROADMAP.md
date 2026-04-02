# VecTrace Roadmap

## Principle

Ship the user-facing MVP in Python first. Introduce Rust only for measured bottlenecks.

## Phase 1 (Weeks 1-4): Python MVP

### Scope
- `vectrace init`: initialize SQLite lineage schema.
- `vectrace trace`: resolve vector -> chunk -> document lineage.
- `vectrace report`: generate HTML lineage report.
- `vectrace connect`: verify Qdrant connectivity and collection.
- Qdrant wrapper for tracked upserts (`TrackedQdrant`).

### Exit Criteria
- Install + first run in under 2 minutes.
- End-to-end demo from ingest to report in under 30 seconds (small dataset).
- Test suite green in CI (`unittest`).

## Phase 2 (Weeks 5-6): Reliability + Benchmarks

### Scope
- Add benchmark commands and fixtures for ingest/query/report timings.
- Add regression tests for pipeline-run status correctness and idempotent ingest.
- Add migration support for schema changes (`schema_version` table).

### Exit Criteria
- Publish benchmark baseline for:
  - vectors/sec ingest (10k, 100k points)
  - p95 `trace` latency (single vector lookup)
  - peak RSS during ingest jobs

## Phase 3 (Rust Triggered Only by Data)

Rust work starts only if **both** conditions are met:
- A measured bottleneck is in VecTrace compute path (not external API/network latency).
- Python misses agreed target by at least 30% on production-like workloads.

### Candidate Rust Components
- Large-scale vector diff/comparison engine.
- Bulk transform/normalization pipeline.
- Hashing/similarity kernels for large batch processing.

### Integration Strategy
- Keep CLI and public API Python-first.
- Add Rust as optional internal core (`PyO3`) behind stable Python interfaces.
- Keep data model + schema unchanged across Python/Rust paths.

## Performance Gates

Before any Rust rewrite, capture:
- workload definition (dataset shape, hardware, concurrency)
- current Python baseline
- target SLO (throughput/latency/memory)
- expected ROI from rewrite

If no clear ROI, do not rewrite.
