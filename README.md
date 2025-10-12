# synth-data-gen

Synthetic-data-gen is an end‑to‑end toolkit for profiling production JSON, synthesising privacy‑preserving lookalike datasets, and validating the results. It is built around a Typer CLI and a local LLM orchestration layer so you can infer complex rules (regexes, semantics, PK/FK graphs, temporal trends) while keeping sensitive data on your machine.

The project is designed for teams that need realistic test fixtures without shipping real customer data. It ingests raw JSON, produces an interpretable ruleset, generates synthetic JSONL output with streaming + checkpointing, and validates fidelity/privy metrics with an HTML report.

## Key Features

- **LLM-assisted profiling** – Optional GLM‑4.5‑Air integration adds regex/text rules, anomaly narratives, semantic labels, and confidence scores alongside heuristic stats.
- **Ruleset synthesis** – Converts profile metadata into an editable YAML ruleset capturing types, enums, regexes, temporal hints, FK graphs, and plugin hooks.
- **Privacy enforcement** – Automatic PII detection, quasi-identifier generalisation, numeric bucketisation, and differential privacy noise metadata.
- **Deterministic generation** – Stream synthetic JSONL while respecting uniqueness/PK constraints, with checkpoint resume, RNG caching, and plugin overrides for custom fields.
- **Temporal/correlation analytics** – Profiles correlation drift, autocorrelation, inter-arrival stats, KS distances, and renders comparison tables in the report.
- **Rich CLI ergonomics** – Progress bars with ETA, resume-aware file handling, streaming readers for large JSON, and consistent cache directories.
- **HTML reporting** – Converts validation JSON into a dashboard covering schema deltas, rule violations, correlation drift, temporal alignment, and privacy posture.
- **Test coverage** – Extensive unit and integration tests for generator enhancements, analysis modules, pipeline flow, and checkpoint resumption.

## Setup

Requirements:

- Python 3.13+
- `uv` (recommended) or `pip` for dependency management
- A running local LLM endpoint if you intend to enable `--use-llm` (defaults assume llama.cpp at `http://localhost:8080/v1/chat/completions`)

1. **Clone & bootstrap**
   ```bash
   git clone <repo-url> synth-data-gen
   cd synth-data-gen
   uv sync  # or python -m venv .venv && source .venv/bin/activate && pip install -r requirements
   ```

2. **Activate virtualenv (if not using `uv sync`’s auto-env)**
   ```bash
   source .venv/bin/activate
   ```

3. **Run tests (optional sanity check)**
   ```bash
   python -m unittest discover
   ```

4. **Configure LLM endpoint (optional)**
   - Start `llama.cpp` or your preferred service exposing a Chat Completions API compatible with `http://localhost:8080/v1/chat/completions`.
   - Update `synth/llm.py` configuration if your endpoint differs (`endpoint`, `model`, `timeout`, `temperature`, `top_p`).

## Usage Guide

All commands live under `python -m synth.cli` (or `synth` if you expose a console entry point). The standard pipeline is:

1. **Profile**
   ```bash
   python -m synth.cli profile examples/sample.json \
       --output tmp/profile.json \
       --chunk-size 128 \
       --use-llm \
       --cache-dir .cache/llm
   ```
   - Produces structural & statistical metadata (types, enums, regex hints, anomalies, temporal summaries, PK/FK candidates).

2. **Synthesize rules**
   ```bash
   python -m synth.cli rules tmp/profile.json \
       --output tmp/ruleset.yaml \
       --privacy
   ```
   - Generates YAML rules, applying privacy hardening by default (`--no-privacy` to skip). Edit this file if you want to tweak generators or add plugin overrides.

3. **Generate synthetic data**
   ```bash
   python -m synth.cli synthesize tmp/profile.json \
       --ruleset-path tmp/ruleset.yaml \
       --output tmp/synthetic.jsonl \
       --count 5000 \
       --seed 1234 \
       --checkpoint-path tmp/generation.ckpt \
       --checkpoint-interval 500
   ```
   - Streams JSONL with a progress bar (elapsed + ETA).
   - Checkpointing enables resume after interruptions; the CLI truncates partially written output before resuming.

4. **Validate**
   ```bash
   python -m synth.cli validate tmp/synthetic.jsonl \
       --source-path examples/sample.json \
       --ruleset-path tmp/ruleset.yaml \
       --report-path tmp/validation.json
   ```
   - Reprofiles source vs synthetic data, checks rules, computes KS/PSI deltas, correlation drift, temporal alignment, k-anonymity, and DP metadata. Output is JSON by design.

5. **Render report**
   ```bash
   python -m synth.cli report tmp/validation.json --output tmp/report.html
   ```
   - Turns validation JSON into an HTML dashboard.

### Additional Flags & Tips

- `--cache-dir` – supply a persistent cache for LLM responses and profile results.
- `--default-array-cap` – fallback for array lengths when no metadata is present (default 5).
- `--use-llm` – toggles the LLM-enhanced profiling pass.
- `--checkpoint-path` / `--checkpoint-interval` – control resume behaviour; intervals default to 1000.
- Resume-friendly generation automatically truncates the output file to the saved checkpoint count before appending new records.
- Validation outputs JSON even when `--report-path` is supplied; pipe that JSON to the `report` subcommand to get HTML.
- When using plugins, register factories via `synth/plugin_registry.py` or place them under `plugins/`.

## Contributing & Testing

- Run `python -m unittest discover` before pushing changes.
- Add targeted tests alongside features (see `tests/` for patterns).
- Keep README instructions up to date when adding CLI flags or configuration defaults.

## Roadmap Ideas

- Replace Pandas `.view` usage with `.astype` (see validation warnings).
- Expand report visuals with Plotly charts.
- Add performance benchmarks for large datasets.
