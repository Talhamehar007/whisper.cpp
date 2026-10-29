# Changelog

All notable changes and technical work sessions for this project are recorded in this file.
Every entry represents an isolated, standalone piece of work documented with in-depth technical breakdowns and git telemetry.

---

## Table of Contents & Work Log Index

- **Total Separate Work Sessions Logged:** 1
- **Total Git Commits Created Across All Entries:** 1

| Entry # | Date & Time | Work / Change Title | Agent & Model | Commits Count & Hashes | Jump Link |
| :---: | :---: | :--- | :--- | :--- | :---: |
| Entry #1 | 2026-10-29 08:05:18 GMT | Migrate Core ML tools to uv, compile ANE support & execute M4 benchmarks | Antigravity (agy) (gemini-2.5-pro) | 1 Commit (c4d9461b) | [Jump to Entry #1](#entry-1) |

---

<a id="entry-1"></a>
## [Entry #1] 2026-10-29 08:05:18 GMT — Migrate Core ML tools to uv, compile ANE support & execute M4 benchmarks

- **Date & Exact Timestamp:** `2026-10-29 08:05:18 GMT`
- **Executed By Agent:** Antigravity (agy) | Model: `gemini-2.5-pro`
- **Category:** `Added` | `Changed`
- **Commits Made in This Specific Work:** 1 Commit (`c4d9461b`)
  - Commit `c4d9461b` (`2026-10-29 08:05:56 GMT`): `build(coreml): migrate conversion script to uv and add M4 benchmarks`
- **Files Touched Summary:** 6 files (`models/generate-coreml-model.sh`, `pyproject.toml`, `.python-version`, `uv.lock`, `BENCHMARKS.md`, `CHANGELOG.md`)

### 1. In-Depth Executive Summary of Work
Resolved dependency failures when running `./models/generate-coreml-model.sh` by introducing modern Python package management with `uv`. Configured a reproducible Python 3.12 virtual environment providing `torch`, `coremltools`, `openai-whisper`, and `ane-transformers`. Patched the model generation shell script to dynamically detect and prioritize `uv run python`. Compiled `whisper.cpp` with Apple Neural Engine acceleration enabled (`-DWHISPER_COREML=1 -DWHISPER_COREML_ALLOW_FALLBACK=1`), successfully generating compiled Core ML encoder bundles (`.mlmodelc`) across all 5 standard Whisper model tiers (`tiny.en`, `base.en`, `small.en`, `medium.en`, `large-v3-turbo`).

Conducted an exhaustive latency, throughput, and power efficiency benchmark on Apple Silicon M4 using a long-form audio sample (`samples/gb0.wav`, 127.36 seconds / 2 minutes 7 seconds). Compared ANE (Core ML + Metal) against native Metal GPU and pure CPU inference across all models, capturing real-time speed multipliers (up to 79.4x real-time), Real-Time Factors (RTF), encoder times, and decoder times. Documented all findings and reproduction instructions in `BENCHMARKS.md`.

### 2. Exhaustive Change-by-Change & File Breakdown
- **`models/generate-coreml-model.sh`** (`+10 / -2 lines`):
  - **Symbols Modified:** Shell runtime variable `$PYTHON` detection logic.
  - **Detailed Breakdown of Changes:** Added dynamic check for the `uv` package manager; if `uv` is present on `$PATH`, `PYTHON` is set to `uv run python` with fallback to `python3` or user-defined `$PYTHON`. Replaced direct `python3` invocations with `$PYTHON`.
  - **Reason & Technical Impact:** Prevents `ModuleNotFoundError` on systems where system Python lacks ML frameworks; ensures all conversion runs occur in the hermetic `uv` virtual environment.
- **`pyproject.toml`** (`+16 lines`):
  - **Detailed Breakdown of Changes:** Initialized Python project metadata with dependencies: `torch>=2.2.0`, `coremltools>=9.0`, `openai-whisper>=20250625`, `ane-transformers>=0.1.1`, `transformers>=4.30.0`, `huggingface-hub>=0.20.0`, and `numpy`.
  - **Reason & Technical Impact:** Standardizes Core ML conversion tooling dependencies with deterministic resolution.
- **`.python-version`** (`+1 line`):
  - **Detailed Breakdown of Changes:** Pin Python version to `3.12`.
- **`uv.lock`** (`+locked dependencies`):
  - **Detailed Breakdown of Changes:** Hermetic lockfile freezing exact wheel releases and hashes.
- **`BENCHMARKS.md`** (`+160 lines`):
  - **Detailed Breakdown of Changes:** Comprehensive benchmark report detailing M4 hardware specifications, master comparison table across all 5 models and 3 execution modes, model-by-model analysis, architectural insights, and setup instructions.

### 3. Verification & System Impact
- **Tests & Commands Run:**
  - `uv sync` — Verified dependency resolution and virtual environment creation.
  - `./models/generate-coreml-model.sh <model>` — Compiled `.mlmodelc` bundles for `tiny.en`, `base.en`, `small.en`, `medium.en`, and `large-v3-turbo`.
  - `cmake -B build -DWHISPER_COREML=1 -DWHISPER_COREML_ALLOW_FALLBACK=1 && cmake --build build -j` — Successfully compiled Core ML-enabled binaries.
  - Automated benchmark harness on `samples/gb0.wav` (127.36s) across all 5 models and 3 execution modes (ANE, Metal GPU, CPU).
- **Breaking Changes / Side Effects:** None. Native C/C++ build remains fully backward-compatible. Python conversion script seamlessly falls back to `python3` if `uv` is unavailable.

---
