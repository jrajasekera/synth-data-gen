# Synthetic JSON Generator — LLM-Centric Design (Enhanced)

> **Goal:** Automatically learn patterns from production JSON and generate **production-like**, **schema-identical** synthetic datasets for development and testing — using a **local LLM** (GLM-4.5-Air) served by **llama.cpp**, with enhanced statistical rigor, privacy controls, and extensibility.

---

## Why this exists

- You often can't use prod data directly for tests. Manual redaction/mocking is slow and error-prone.
- This tool ingests a representative prod JSON sample, **infers structure + rules**, and emits a synthetic twin that:
  - preserves **schema fidelity** (types, nesting, arrays),
  - approximates **distributions and correlations**,
  - respects **business constraints** (PK/FK, uniqueness, regex formats),
  - captures **temporal patterns** (seasonality, trends, cycles),
  - supports **deterministic** regeneration via seeds,
  - provides **privacy guarantees** (differential privacy, k-anonymity options),
  - and enables **incremental generation** for massive datasets.

Because you're running the LLM **locally** (no data leaves your machine), we can let the model analyze **raw JSON** directly for deeper pattern mining and text grammar extraction.

---

## High-level architecture

```
Input JSON
   │
   ▼
A) LLM-Driven Profiler  ──► JSON Schema + Rich Profile (stats + rules)
   │                           - streaming map-reduce with checkpointing
   │                           - PII detection and anomaly flagging
   ▼
B) Rule Synthesis        ──► ruleset.yaml (human-editable)
   │                           - structural rules, distributions, FDs, regex, PK/FK
   │                           - temporal patterns, correlation graphs
   │                           - privacy constraints
   ▼
C) Synthetic Generator   ──► synthetic.json
   │                           - classic samplers + LLM for hard text/conditionals
   │                           - streaming output for large datasets
   │                           - plugin architecture for custom generators
   ▼
D) Validators & Reports  ──► report.html (fit + integrity + privacy)
   │                           - statistical tests, privacy metrics
   │                           - quality scoring and recommendations
   ▼
E) Incremental Updates   ──► delta.json (append-only synthetic data)
```

### Guarantees & knobs

- **Schema fidelity:** JSON Schema (draft-07) validation with extensibility.
- **Statistical realism:** per-field distributions, conditional distributions, correlation preservation (Gaussian copulas, Bayesian networks).
- **Temporal patterns:** seasonality, trends, day-of-week/hour-of-day rhythms, autocorrelation.
- **Business rules:** uniqueness, composite keys, FK graphs, conditional presence, regex formats, checksums (e.g., Luhn).
- **Determinism:** global seed + per-field/row seed; batched LLM calls are seed-conditioned with caching.
- **Privacy posture:** local inference with optional differential privacy, k-anonymity checks, PII detection, verbatim-match guard, and rare-category caps.
- **Performance:** streaming I/O, parallel generation, checkpointing, LLM response caching.

---

## Current Status (as of October 2025)

Implemented:
- Typer-based CLI covering profiling, synthesis, rules export, validation, and report rendering with configurable array caps.
- Profiler with heuristic + optional LLM map-reduce pass capturing types, enum candidates, regex hints, semantic labels, anomalies, PK candidates, and array length statistics.
- Rule synthesis that persists inferred constraints (regex, semantic type, array metadata, PII likelihood) with privacy sanitisation for sensitive enums and optional LLM refinements.
- Synthetic generator supporting recursive object/array traversal, regex-aware text synthesis, Faker semantic fillers, plugin overrides, and privacy-aware rule adjustments.
- Validation that reprofiles source/synthetic data, compares type/coverage/numeric/array metrics, and renders an HTML report with delta tables.
- Integration tests covering the profile→rules→generate pipeline, regex compliance, plugin hooks, nested arrays, and array-cap enforcement.

Outstanding (high priority):
- Expanded LLM capabilities (confidence scoring, anomaly narratives, conditional/text rule authoring) to further reduce heuristic fallbacks.
- Advanced rule synthesis (composite keys, FK graphs, functional dependencies, correlation/temporal models, conditional constraints).
- Rich synthetic generation (conditional logic, streaming/delta modes, semantic templates, scenario presets) and tighter plugin ecosystem.
- Stat/Privacy validation (KS/χ²/PSI/Wasserstein), k-anonymity/DP checks, privacy reporting, and visual dashboards.
- Temporal/correlation modules, incremental updates, caching strategy, configuration ergonomics, and comprehensive documentation.

---

## Detailed implementation plan

### 0) Repo skeleton & tooling

```
synth-data-gen/
├─ pyproject.toml                 # uv/poetry; pin numpy/scipy/jsonschema/typer
├─ synth/
│  ├─ cli.py                      # typer CLI with rich progress bars
│  ├─ io.py                       # JSON streaming & chunking with checkpointing
│  ├─ llm.py                      # llama.cpp client + response cache + fallbacks
│  ├─ profile.py                  # LLM map-reduce orchestrator + numeric verification
│  ├─ rules.py                    # rule synthesis from profile -> ruleset.yaml
│  ├─ generate.py                 # samplers + batched LLM text generation
│  ├─ validate.py                 # schema, PK/FK/unique, KS/χ²/PSI, privacy checks
│  ├─ report.py                   # HTML report (Jinja2) with charts and recommendations
│  ├─ privacy.py                  # PII detection, k-anonymity, differential privacy
│  ├─ temporal.py                 # time series patterns, seasonality detection
│  ├─ correlations.py             # copulas, Bayesian networks, correlation graphs
│  ├─ plugins.py                  # plugin loader for custom generators/validators
│  ├─ cache.py                    # LLM response cache with TTL and invalidation
│  └─ utils.py                    # RNG seeds, JSONPath helpers, regex tools
├─ plugins/
│  ├─ generators/                 # custom field generators
│  └─ validators/                 # custom validation rules
├─ templates/
│  ├─ report.html.j2              # HTML report template
│  └─ field_templates.yaml        # common text templates library
├─ examples/
│  ├─ sample.json
│  ├─ ruleset.yaml
│  └─ scenarios/
│     ├─ black_friday.yaml
│     └─ maintenance_window.yaml
├─ tests/
│  ├─ unit/
│  ├─ integration/
│  └─ property_based/
├─ bench/
│  ├─ datasets/
│  └─ benchmark.py
└─ README.md
```

**Key deps**
- `typer`, `rich`, `pydantic`, `jsonschema`, `jsonpath-ng`, `genson` (schema inference)
- `numpy`, `pandas`, `scipy`, `scikit-learn`, `statsmodels` (stats & tests)
- `jinja2`, `markdown-it-py`, `plotly` (interactive report charts)
- `faker` (templated text fallback)
- `httpx` (llama.cpp server client with retry logic)
- `diskcache` (persistent LLM response cache)
- `presidio-analyzer` (PII detection, optional)
- `hypothesis` (property-based testing)

> Package management: **uv** (Ubuntu).

```bash
uv init synth-data-gen
uv add typer rich pydantic jsonschema jsonpath-ng genson numpy pandas scipy scikit-learn statsmodels jinja2 markdown-it-py plotly faker httpx diskcache presidio-analyzer hypothesis
```

---

### 1) LLM server (llama.cpp) — GLM-4.5-Air with optimizations

Start a local inference server with performance tuning:

```bash
# Example: start llama.cpp server with GLM-4.5-Air GGUF
./llama-server \
  -m /models/GLM-4.5-Air.Q4_K_M.gguf \
  -c 8192 \
  -ngl 99 \
  --host 127.0.0.1 \
  --port 8080 \
  --slots 8 \
  --parallel 8 \
  --timeout 120 \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --cont-batching
```

**Client config (`synth/config.toml`):**
```toml
[llm]
base_url = "http://127.0.0.1:8080"
model = "GLM-4.5-Air"
max_tokens = 4096
temperature = 0.2
top_p = 0.9
seed = 42
timeout = 60
max_retries = 3
retry_backoff = 2.0

[llm.cache]
enabled = true
ttl = 86400  # 24 hours
max_size_mb = 1024

[llm.fallback]
enabled = true
strategies = ["faker", "regex_sampler", "template"]
```

`llm.py` should implement:
- `complete_json(prompt, schema_hint, seed)` — returns parsed JSON; retries with exponential backoff until valid.
- `batch_generate_text(field_path, rules, start_index, count, seed)` — returns `list[str]` with quality scoring.
- `get_cached(prompt_hash)` / `set_cached(prompt_hash, response)` — persistent cache with TTL.
- `estimate_tokens(text)` — token usage estimation for cost tracking.
- **Fallback cascade:** LLM → template → faker → regex_sampler on failure.

---

### 2) LLM-Driven Profiler (streaming map-reduce with checkpointing)

**Streaming & chunking**: for datasets >100MB, use streaming JSON parser with configurable chunk size (50–200 items). Save checkpoints every N chunks to enable resumption.

**Incremental profiling**: support `--update` mode to profile only new records and merge with existing profile.

**Map prompt** (strict JSON output with structured generation):

_System_
```
You analyze raw JSON to infer schema, constraints, patterns, and temporal relationships.
Return ONLY JSON matching OUTPUT_SCHEMA exactly. Assign confidence scores (0..1) to uncertain inferences.
Flag potential PII fields. Detect anomalies and outliers.
```

_User_
```
OUTPUT_SCHEMA:
{
  "field_summaries": {
    "<jsonpath>": {
      "types": [{"name":"string|number|boolean|datetime|object|array|null","confidence":0..1}],
      "required_rate": 0..1,
      "enum_candidates": [{"value":"...", "count":int, "confidence":0..1}],
      "regex_candidates": [{"pattern":"...", "support":int, "generality":0..1}],
      "length": {"min":int,"p25":int,"p50":int,"p75":int,"p95":int,"max":int},
      "numeric": {"min":num,"p5":num,"p25":num,"p50":num,"p75":num,"p95":num,"max":num,"std":num,"skew":num,"kurtosis":num},
      "datetime": {"min":"iso","max":"iso","by_weekday":[int],"by_hour":[int],"seasonality":{"detected":bool,"period":str}},
      "text_features": {"avg_word_count":float,"common_tokens":[str],"sentiment_hint":str},
      "pii_likelihood": 0..1,
      "anomalies": [{"value":"...","reason":str}]
    }
  },
  "keys": {
    "pk_candidates": [{"paths":["$.orders[].id"], "uniqueness":0..1, "confidence":0..1}],
    "fk_candidates": [{"parent":"$.customers[].id","child":"$.orders[].customer_id","coverage":0..1,"confidence":0..1}],
    "composite_key_candidates": [{"paths":["$.store_id","$.date"],"confidence":0..1}]
  },
  "functional_dependencies": [
    {"lhs":["$.channel"],"rhs":["$.payment_method"],"support":0..1,"confidence":0..1,"conditional":false}
  ],
  "temporal_patterns": [
    {"field":"$.created_at","trend":"increasing|decreasing|stable","seasonality":"daily|weekly|monthly|none","autocorrelation":0..1}
  ],
  "notes": ["..."]
}

DATA_CHUNK:
<literal JSON array or object>
CHUNK_ID: {chunk_id}
SEED: {seed}
```

**Reduce prompt**: merge map outputs; deduplicate; rank candidates by `support × confidence × generality`; prefer simpler regexes and higher-level patterns.

**Ground-truth verification**: after reduce, compute exact statistics from raw data (quantiles, frequencies, correlations, temporal autocorrelations) and **overwrite** any LLM-reported numbers. Save:
- `schema.json` (draft-07, via `genson` + fixups + semantic annotations)
- `profile.json` (stats, distributions, correlations, temporal patterns, PII flags)
- `rule_hypotheses.json` (machine-readable candidates with confidences)
- `anomalies.json` (detected outliers and unusual patterns)

**New: PII detection**
- Run regex patterns + presidio-analyzer on string fields.
- Flag fields as `email`, `phone`, `ssn`, `credit_card`, `name`, `address`, etc.
- Recommend anonymization strategies in `ruleset.yaml`.

---

### 3) Rule Synthesis → `ruleset.yaml` (enhanced)

**YAML schema (v2):**
```yaml
version: 2
metadata:
  source: examples/sample.json
  created_at: 2025-01-15T10:30:00Z
  profiled_rows: 10000
  llm_model: GLM-4.5-Air

globals:
  seed: 42
  temperature: 0.2
  default_n: 10000
  parallel_workers: 4

shape:
  $.orders[]:
    length:
      distribution: empirical
      params: {quantiles: [0,5,25,50,75,95,100], values: [1,1,2,3,7,15,20]}
  $.orders[].items[]:
    length:
      distribution: poisson
      params: {lambda: 2.3}

fields:
  $.orders[].id:
    type: string
    semantic_type: uuid
    unique: true
    generator: uuid_v4
    privacy: {pii: false}

  $.orders[].created_at:
    type: datetime
    generator: temporal_rhythmic
    params:
      trend: stable
      weekday_profile: [0.08, 0.12, 0.15, 0.18, 0.22, 0.15, 0.10]  # Mon-Sun
      hour_profile: [0.01, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.15, 0.14, 0.12, 0.10, 0.08, 0.05, 0.03, 0.01]
      seasonality: weekly
      autocorrelation: 0.15

  $.orders[].amount:
    type: number
    generator: quantile_sample
    params: {quantiles: [0,5,25,50,75,95,100], values: [5.0, 12.5, 28.0, 45.0, 78.0, 150.0, 500.0]}
    constraints:
      min: 0.01
      max: 10000.0
      decimals: 2

  $.orders[].status:
    type: enum
    generator: categorical
    params:
      values: [PENDING, PAID, FAILED, REFUNDED]
      probs:  [0.22, 0.63, 0.10, 0.05]
    transitions:  # Markov chain for state evolution
      PENDING: {PAID: 0.7, FAILED: 0.2, REFUNDED: 0.1}
      PAID: {REFUNDED: 0.05, PAID: 0.95}

  $.customer.email:
    type: string
    semantic_type: email
    constraints:
      regex: "^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$"
      length: {min: 6, max: 64}
    generator: template
    params:
      template: "{{first}}.{{last}}{{num}}@{{domain}}"
      domains: ["example.test", "demo.com", "test.org"]
    privacy:
      pii: true
      anonymization: template  # use synthetic template instead of LLM
      k_anonymity: 5

  $.customer.phone:
    type: string
    semantic_type: phone
    generator: faker
    params: {method: "phone_number", locale: "en_US"}
    privacy:
      pii: true
      anonymization: faker

  $.customer.notes:
    type: string
    generator: llm_text
    params:
      constraints: {min_length: 10, max_length: 200, style: "customer_feedback"}
      temperature: 0.3
      max_batch: 128
    fallback: {generator: template, params: {template: "Order notes for reference #{{num}}"}}

relationships:
  - type: foreign_key
    parent: "$.customers[].id"
    child:  "$.orders[].customer_id"
    cardinality: {min: 0, max: 50, mean: 3.2}
  - type: functional_dep
    lhs:   ["$.customer.country"]
    rhs:   ["$.currency"]
    confidence: 0.98

correlations:
  - fields: ["$.order.amount", "$.order.items[].length"]
    method: gaussian_copula
    correlation: 0.67
  - fields: ["$.customer.age", "$.order.category"]
    method: categorical_association
    cramers_v: 0.43

privacy:
  enable_verbatim_guard: true
  rare_category_threshold: 0.01
  k_anonymity:
    enabled: true
    k: 5
    quasi_identifiers: ["$.customer.zip", "$.customer.age_range"]
  differential_privacy:
    enabled: false
    epsilon: 1.0
    mechanism: laplace

plugins:
  generators:
    - path: plugins/generators/custom_sku.py
      fields: ["$.product.sku"]
  validators:
    - path: plugins/validators/business_logic.py
```

**Builder logic enhancements**:
- Map each field to optimal generator based on semantic type and complexity.
- Encode temporal patterns (trends, seasonality) for datetime fields.
- Store correlation matrices and dependency graphs.
- Auto-generate privacy recommendations for detected PII.
- Support plugin references for custom business logic.

---

### 4) Synthetic Generator (streaming & parallel)

**Order of operations**
1) **Shape**: object presence, optional keys, array lengths (sampled from learned empirical/Poisson).
2) **Temporal anchors**: establish time range and generate datetime fields with proper sequencing.
3) **Primary keys**: UUID/sequence/composite; maintain registries for FK resolution.
4) **Correlated dimensions**: sample numeric/categorical fields respecting learned correlations (copulas/Bayesian nets).
5) **Referential integrity**: sample FK values from parent registries honoring learned cardinalities.
6) **Conditional fields**: apply functional dependencies and conditional presence rules.
7) **Text fields**: choose `template → faker → regex_grammar → LLM` cascade with quality checks.
8) **Privacy transforms**: apply k-anonymity, differential privacy noise, PII anonymization.

**Streaming output for large datasets**
```python
def generate_stream(rules, n, output_path, checkpoint_interval=1000):
    with jsonl_writer(output_path) as writer:
        for batch in range(0, n, checkpoint_interval):
            objs = generate_batch(rules, min(checkpoint_interval, n - batch))
            writer.write_batch(objs)
            save_checkpoint(batch, registries)
```

**Parallel generation with worker pools**
```python
def generate_parallel(rules, n, workers=4):
    with ProcessPoolExecutor(max_workers=workers) as executor:
        chunk_size = n // workers
        futures = [
            executor.submit(generate_batch, rules, chunk_size, seed=seed+i)
            for i in range(workers)
        ]
        return merge_results([f.result() for f in futures])
```

**LLM text batching with caching**
- Generate 64–256 rows per prompt for throughput.
- Check cache first: `cache_key = hash(field_path, rules, seed, indices)`.
- Prompt includes `{field_path, start_index, count, seed, constraints, examples}` and must return a **JSON array of strings** only.
- Post-validate against regex/length; repair with same seed if needed.
- On repeated failures, fall back to simpler generators.

**Quality scoring for generated text**
```python
def score_text_quality(text, rules):
    scores = {
        "length_match": check_length(text, rules),
        "regex_match": check_regex(text, rules),
        "coherence": estimate_coherence(text),
        "diversity": check_uniqueness(text, generated_set)
    }
    return weighted_average(scores)
```

**Scenario overlays (enhanced)**
```yaml
scenarios:
  black_friday:
    description: "Simulate Black Friday traffic spike"
    temporal_range: ["2025-11-29T00:00:00", "2025-11-29T23:59:59"]
    modifications:
      $.orders[].created_at:
        hour_profile: {scale: {8..23: 2.5}}
      $.orders[].amount:
        distribution: shift
        params: {mean: +25%, std: +10%}
      $.orders[].status:
        probs: [0.35, 0.50, 0.12, 0.03]
  
  maintenance_window:
    description: "System maintenance causing reduced orders"
    temporal_range: ["2025-12-15T02:00:00", "2025-12-15T06:00:00"]
    modifications:
      $.orders[]:
        length: {distribution: constant, value: 0}
```

---

### 5) Validation & Reporting (comprehensive)

**Structural & business checks**
- JSON Schema validation (draft-07)
- Unique constraints & composite keys
- FK reachability (all children reference generated parents)
- Conditional rules (JSONPath + predicate DSL)
- Plugin validators for custom business logic

**Statistical fit tests**
- **Continuous**: Kolmogorov-Smirnov, Anderson-Darling
- **Categorical**: χ² test, G-test
- **Shift metrics**: PSI, Jensen-Shannon divergence, Wasserstein distance
- **Correlations**: Δ heatmap (Pearson for numeric, Cramér's V for categorical)
- **Temporal**: autocorrelation function comparison, seasonality decomposition
- **Distribution comparison**: Q-Q plots, P-P plots

**Privacy checks**
- **k-anonymity**: verify quasi-identifier combinations appear ≥k times
- **l-diversity**: check sensitive attribute diversity within equivalence classes
- **Differential privacy**: validate noise injection meets ε-budget
- **Verbatim guard**: ensure no exact matches to rare prod values
- **PII leakage**: scan for accidentally preserved PII patterns

**Quality metrics**
```python
quality_score = {
    "schema_compliance": 1.0,  # pass/fail
    "statistical_fit": 0.92,   # weighted avg of p-values
    "correlation_preservation": 0.88,  # Frobenius norm of Δ
    "temporal_fidelity": 0.95,  # autocorrelation match
    "privacy_score": 1.0,  # k-anonymity + l-diversity pass rate
    "business_rules": 0.98,  # constraint satisfaction rate
    "overall": 0.93  # weighted composite
}
```

**Report (`report.html` with interactivity)**
- Executive summary with quality score badges
- **Statistical section**:
  - Per-field fit charts (histograms, Q-Q plots)
  - Correlation heatmap comparison (real vs synthetic)
  - Temporal pattern comparison (ACF, seasonal decomposition)
- **Integrity section**:
  - Rule coverage & violations table
  - FK/PK integrity summary with graph visualization
  - Uniqueness constraint pass rates
- **Privacy section**:
  - k-anonymity assessment
  - PII leakage scan results
  - Verbatim match report
- **Performance section**:
  - Generation timings (profiling, generation, validation)
  - LLM usage stats (tokens, cache hit rate, cost estimate)
  - Memory and throughput metrics
- **Recommendations**:
  - Auto-generated suggestions to improve fit or address issues
  - Parameter tuning hints
  - Warning flags for concerning patterns

---

## CLI design (enhanced)

```bash
# 1) Profile real data with streaming and checkpoints
synth profile examples/sample.json \
  --out ruleset.yaml \
  --schema-out schema.json \
  --profile-out profile.json \
  --anomalies-out anomalies.json \
  --model "GLM-4.5-Air" \
  --server "http://127.0.0.1:8080" \
  --seed 42 \
  --streaming \
  --chunk-size 100 \
  --checkpoint-interval 1000

# 2) Update existing profile with new data (incremental)
synth profile examples/new_data.json \
  --update ruleset.yaml \
  --merge-strategy weighted

# 3) Generate N synthetic rows with streaming output
synth generate \
  --rules ruleset.yaml \
  --n 50000 \
  --out synthetic.jsonl \
  --seed 42 \
  --streaming \
  --parallel 4 \
  --checkpoint-interval 5000 \
  --progress

# 4) Generate with scenario overlay
synth generate \
  --rules ruleset.yaml \
  --scenario examples/scenarios/black_friday.yaml \
  --out synthetic_bf.json \
  --seed 1337

# 5) Validate realism, integrity, and privacy
synth validate \
  --real examples/sample.json \
  --synthetic synthetic.json \
  --rules ruleset.yaml \
  --report report.html \
  --privacy-checks \
  --statistical-tests all

# 6) Dry run (profile + generate sample + validate) without full generation
synth dryrun \
  --input examples/sample.json \
  --n 1000 \
  --report dryrun_report.html

# 7) Interactive rule editor (TUI)
synth edit ruleset.yaml

# 8) Export masked examples for unit tests
synth export-examples \
  --rules ruleset.yaml \
  --n 10 \
  --out tests/fixtures/

# 9) Benchmark performance
synth benchmark \
  --rules ruleset.yaml \
  --sizes 1000,10000,100000 \
  --report bench_report.html
```

**Environmental config**
- `SYNTH_LLM_BASE_URL`, `SYNTH_LLM_MODEL`, `SYNTH_SEED`
- `SYNTH_CACHE_DIR`, `SYNTH_LOG_LEVEL`, `SYNTH_PARALLEL_WORKERS`

---

## Determinism strategy (with LLM caching)

- Use a **central RNG** for all non-LLM generators. Store `globals.seed` in YAML.
- For each LLM call, include a **derivative seed**: `seed_f = hash64(global_seed, field_path, batch_id, start_index)`.
- Fix `temperature` low (e.g., 0.2) and constrain outputs with regex/length.
- **Cache LLM responses** with key: `hash(prompt, seed, temperature, constraints)`.
- Implement **retry-on-violation** with the same seed + stricter constraints (max 3 retries).
- For streaming generation, checkpoint RNG state along with data.

**Cache invalidation**: TTL-based (default 24h) + manual purge command.

---

## Security & privacy posture (local model with formal guarantees)

### Local inference
- **No network calls** beyond your llama.cpp endpoint (all data stays local).
- Optional audit log of all LLM prompts (disabled by default for performance).

### PII protection
- **Automatic detection**: regex + ML-based (presidio-analyzer) for common PII types.
- **Anonymization strategies**:
  - Templates for emails, phones, addresses
  - Faker for names, locations
  - Generalization for numeric quasi-identifiers (age → age_range)
  - Suppression for rare sensitive values

### Privacy guarantees
- **k-anonymity**: ensure each combination of quasi-identifiers appears ≥k times.
- **l-diversity**: ensure sensitive attributes have ≥l distinct values per equivalence class.
- **Differential privacy** (optional): add calibrated Laplace/Gaussian noise to numeric aggregates.
- **Verbatim guard**: prevent exact replication of rare strings (Jaro-Winkler threshold).
- **Rare category suppression**: cap low-frequency categorical values (configurable threshold).

**Privacy-utility tradeoff dashboard**: show impact of privacy settings on statistical fidelity.

---

## Sampling algorithms (extended)

### Numeric fields
- **Quantile interpolation**: inverse-CDF from stored quantiles (monotone cubic spline).
- **KDE**: kernel density estimation for smoother tails (Gaussian/Epanechnikov kernels).
- **Parametric**: fit Gamma/Lognormal/Beta distributions where appropriate.
- **Clipping & rounding**: enforce min/max/decimals constraints.

### Categorical fields
- **Smoothed frequencies**: Laplace smoothing to prevent zero probabilities.
- **Temperature scaling**: flatten (temperature >1) or peak (temperature <1) distribution.
- **Rare category handling**: pool infrequent values into "OTHER" category.

### Datetime fields
- **Temporal rhythmic**: Poisson process for daily volume × multinomial for weekday/hour profile.
- **Trend injection**: linear/exponential trend component.
- **Seasonality**: additive/multiplicative seasonal decomposition (daily/weekly/monthly/yearly).
- **Autocorrelation**: AR(1) process for temporal dependencies.
- **Inter-arrival times**: exponential distribution with learned rate.

### Arrays
- **Length distribution**: empirical/Poisson/negative binomial.
- **Element order**: preserve via n-gram transition probabilities (optional).
- **Nested structures**: recursive generation with depth limits.

### Correlated fields
- **Gaussian copula**: rank-transform → multivariate normal → inverse rank-transform.
- **Bayesian network**: learn DAG structure, sample from conditional distributions.
- **Covariance matrix**: Cholesky decomposition for multivariate normal sampling.

### Text fields
1) **Templates** (preferred): regex extraction + slot filling.
2) **Regex grammar**: character classes + token vocab; recursive descent sampler.
3) **Markov chains**: character/word-level n-gram models.
4) **LLM-assisted**: when semantics matter (ticket summaries, notes). Provide constraints, seed, and examples.

---

## Text synthesis strategies (detailed)

### 1. Template extraction (preferred)
```python
def extract_template(examples):
    # Find common structure
    patterns = find_common_patterns(examples)
    # Extract variable slots
    template = create_template(patterns)  # e.g., "Order {{id}} shipped via {{carrier}}"
    return template

# Usage
generator = TemplateGenerator(template="Order {{id}} shipped via {{carrier}}")
text = generator.sample(id=uuid.uuid4(), carrier=random.choice(["UPS", "FedEx"]))
```

### 2. Regex grammar sampling
```python
def sample_from_regex(pattern, max_length=100):
    # Parse regex to AST
    ast = parse_regex(pattern)
    # Generate string matching pattern
    return recursive_sample(ast, max_length)
```

### 3. LLM-assisted generation (batched)

_System_
```
Generate realistic values for field {jsonpath}. Follow ALL constraints strictly.
Return a JSON array of exactly {N} strings. No additional text or comments.
Ensure diversity: avoid repetitive patterns.
```

_User_
```
FIELD: {jsonpath}
SEMANTIC_TYPE: {semantic_type}  # e.g., "customer_feedback", "ticket_summary"
CONSTRAINTS:
  - regex: {pattern}
  - min_length: {min}
  - max_length: {max}
  - style: {style_hint}
  - forbidden_words: [...]

EXAMPLES (from real data):
  - "{example_1}"
  - "{example_2}"
  - "{example_3}"

GENERATION_PARAMS:
  - start_index: {k}
  - count: {N}
  - seed: {seed}

Return JSON array: ["text1", "text2", ...]
```

**Response handling**:
- Parse JSON array.
- Validate each string against constraints.
- Score quality (coherence, diversity, constraint adherence).
- If quality < threshold, retry with stricter prompt or fall back to simpler generator.

---

## Plugin architecture

### Custom generators
```python
# plugins/generators/custom_sku.py
from synth.plugins import GeneratorPlugin

class CustomSKUGenerator(GeneratorPlugin):
    def __init__(self, config):
        self.config = config
    
    def generate(self, rng, constraints):
        # Custom SKU format: XXX-YYYY-ZZZ
        prefix = rng.choice(self.config["prefixes"])
        middle = rng.integers(1000, 9999)
        suffix = rng.choice(self.config["suffixes"])
        return f"{prefix}-{middle:04d}-{suffix}"
    
    def validate(self, value):
        return bool(re.match(r"^[A-Z]{3}-\d{4}-[A-Z]{3}$", value))
```

### Custom validators
```python
# plugins/validators/business_logic.py
from synth.plugins import ValidatorPlugin

class OrderConsistencyValidator(ValidatorPlugin):
    def validate_record(self, record, rules):
        # Check: order.amount = sum(order.items[].price * quantity)
        expected_total = sum(
            item["price"] * item["quantity"]
            for item in record.get("items", [])
        )
        actual_total = record.get("amount", 0)
        return abs(expected_total - actual_total) < 0.01
```

**Loading plugins**:
```python
# In synth/plugins.py
def load_generator_plugin(path, field_path):
    module = import_module(path)
    plugin_class = getattr(module, "Generator")
    return plugin_class(config=rules["fields"][field_path])
```

---

## File formats (extended)

- `schema.json`: JSON Schema draft-07 with semantic annotations
- `profile.json`: machine-readable stats, correlations, temporal patterns
- `rule_hypotheses.json`: ranked candidates with confidence scores
- `anomalies.json`: detected outliers and unusual patterns
- `ruleset.yaml`: source of truth for generation (human-editable)
- `synthetic.json` / `synthetic.jsonl`: output data (JSON or line-delimited)
- `report.html`: interactive validation report with charts
- `privacy_report.json`: detailed privacy audit
- `generation.log`: generation logs with checkpoints

---

## Performance optimizations

### Streaming I/O
- Use `ijson` for incremental JSON parsing (avoid loading full dataset).
- Write output incrementally to avoid memory spikes.
- Support line-delimited JSON (JSONL) for easier streaming.

### Caching
- **LLM response cache**: persistent disk cache (diskcache) with LRU eviction.
- **Profile cache**: reuse profiles across runs unless source data changes.
- **Compiled regex cache**: avoid recompiling patterns.

### Parallel processing
- **Profiling**: parallelize map phase across chunks.
- **Generation**: use process pools to generate independent batches.
- **Validation**: parallelize statistical tests across fields.

### Batch operations
- Batch LLM requests (64–256 rows per call).
- Batch database queries for FK lookups (if applicable).
- Vectorized operations for numeric sampling (numpy).

### Resource limits
- Memory-mapped files for very large datasets.
- Generator checkpointing every N records.
- Graceful degradation: if OOM, reduce batch sizes and retry.

**Benchmarking targets**:
- Profile: 10K rows/sec (streaming, no LLM bottleneck)
- Generate: 5K rows/sec (with LLM text ~5% of fields)
- Validate: 20K rows/sec

---

## Testing strategy

### Unit tests
- Test each generator in isolation with known seeds.
- Validate statistical properties of samplers.
- Check constraint enforcement.

### Integration tests
- End-to-end: profile → generate → validate.
- Test with representative datasets (e-commerce, logs, IoT).
- Verify determinism (same seed → same output).

### Property-based tests (hypothesis)
```python
from hypothesis import given, strategies as st

@given(st.lists(st.floats(min_value=0, max_value=100)))
def test_quantile_sampler_preserves_range(data):
    sampler = QuantileSampler.from_data(data)
    samples = [sampler.sample() for _ in range(1000)]
    assert all(min(data) <= s <= max(data) for s in samples)
```

### Privacy tests
- Verify k-anonymity guarantees hold.
- Test verbatim guard prevents exact matches.
- Validate differential privacy noise calibration.

### Regression tests
- Maintain suite of "golden" datasets.
- Detect statistical drift in generated outputs.

---

## Metrics & monitoring

### Generation metrics
- **Throughput**: rows/sec, fields/sec
- **LLM usage**: tokens in/out, cache hit rate, cost estimate
- **Time breakdown**: profiling, generation, validation (per phase)
- **Memory usage**: peak, average, by phase
- **Checkpoint frequency**: time between saves

### Quality metrics
- **Schema compliance**: 100% (hard requirement)
- **Statistical fit**: per-field p-values (KS, χ²)
- **Correlation preservation**: Frobenius norm ||Σ_real - Σ_synth||
- **Temporal fidelity**: ACF match score
- **Privacy score**: k-anonymity pass rate, l-diversity, ε-DP budget

### Alerts & recommendations
- Warning: statistical fit p-value <0.05 for >10% of fields
- Error: FK integrity <95%
- Info: LLM cache hit rate <50% (consider warming cache)
- Recommendation: increase quantile resolution for skewed distributions

**Dashboard**: real-time generation progress with rich progress bars:
```
Profiling  ████████████████████ 100% 10000/10000 rows [00:45, 220 rows/s]
Generating ████████░░░░░░░░░░░░  45%  4500/10000 rows [02:15, 35 rows/s]
  ├─ LLM cache hits: 78%
  ├─ Memory: 2.4GB / 8GB
  └─ ETA: 00:03:20
```

---

## Roadmap

### MVP (v0.1)
- Streaming map-reduce profiling with GLM-4.5-Air
- Numeric/categorical/datetime generators
- PK/FK/unique validation; χ²/KS tests; basic HTML report
- Deterministic RNG + seeded LLM calls with caching
- CLI with progress bars

### v0.2
- Template/regex grammar extraction
- LLM-assisted text generation with batching and fallbacks
- Scenario overlays
- Interactive HTML report with Plotly charts
- PII detection and basic anonymization

### v0.3
- Gaussian copula for correlated numerics
- Temporal patterns (seasonality, trends, ACF)
- k-anonymity and differential privacy options
- Plugin architecture (custom generators/validators)
- TUI rule editor

### v0.4
- Bayesian network learning for complex correlations
- Incremental profiling and generation
- Parallel processing optimizations
- Advanced privacy audit (l-diversity, t-closeness)
- Export test fixtures

### v1.0
- Web UI for schema/rule visualization and job management
- Pluggable backends (vLLM, Ollama, OpenAI API) with unified interface
- Auto-tuning: suggest optimal parameters based on validation results
- Privacy-utility tradeoff optimizer
- Complete documentation and video tutorials

---

## Example end-to-end session

```bash
# Start llama.cpp server
./llama-server \
  -m /models/GLM-4.5-Air.Q4_K_M.gguf \
  -c 8192 \
  -ngl 99 \
  --port 8080 \
  --slots 8 \
  --parallel 8 \
  --cont-batching

# Profile production data with streaming
uv run synth profile examples/sample.json \
  --out ruleset.yaml \
  --schema-out schema.json \
  --profile-out profile.json \
  --anomalies-out anomalies.json \
  --streaming \
  --progress

# Review and edit rules (optional)
uv run synth edit ruleset.yaml

# Dry run with sample generation
uv run synth dryrun \
  --input examples/sample.json \
  --n 1000 \
  --report dryrun_report.html

# Generate full synthetic dataset
uv run synth generate \
  --rules ruleset.yaml \
  --n 50000 \
  --out synthetic.jsonl \
  --seed 42 \
  --streaming \
  --parallel 4 \
  --progress

# Validate quality and privacy
uv run synth validate \
  --real examples/sample.json \
  --synthetic synthetic.jsonl \
  --rules ruleset.yaml \
  --report report.html \
  --privacy-checks \
  --statistical-tests all

# Open interactive report
xdg-open report.html

# Generate with Black Friday scenario
uv run synth generate \
  --rules ruleset.yaml \
  --scenario examples/scenarios/black_friday.yaml \
  --out synthetic_bf.json \
  --seed 1337 \
  --progress

# Export small test fixtures
uv run synth export-examples \
  --rules ruleset.yaml \
  --n 10 \
  --out tests/fixtures/

# Benchmark performance
uv run synth benchmark \
  --rules ruleset.yaml \
  --sizes 1000,10000,100000 \
  --report bench_report.html
```

---

## Advanced features

### Incremental updates
```bash
# Profile new data and merge with existing rules
synth profile new_data.json \
  --update ruleset.yaml \
  --merge-strategy weighted \
  --weight 0.3  # 30% new data, 70% existing

# Generate only new records (append mode)
synth generate \
  --rules ruleset.yaml \
  --append synthetic.jsonl \
  --n 5000 \
  --seed-offset 50000
```

### A/B testing synthetic data
```bash
# Generate two variants with different seeds
synth generate --rules ruleset.yaml --out variant_a.json --seed 42
synth generate --rules ruleset.yaml --out variant_b.json --seed 1337

# Compare statistical properties
synth compare variant_a.json variant_b.json --report comparison.html
```

### Privacy-utility optimization
```python
# Auto-tune privacy parameters to maximize utility while meeting privacy constraints
optimizer = PrivacyUtilityOptimizer(
    target_k_anonymity=5,
    target_utility=0.90,  # 90% statistical fidelity
    epsilon_budget=1.0    # DP budget
)
optimal_config = optimizer.optimize(ruleset)
```

---

## Troubleshooting guide

### Common issues

**LLM timeouts or failures**
- Reduce batch size: `max_batch: 64 → 32`
- Increase timeout: `timeout: 60 → 120`
- Enable fallback generators: `fallback: {generator: template, ...}`
- Check llama.cpp server logs for OOM or GPU errors

**Poor statistical fit**
- Increase profile sample size
- Use finer quantile resolution: `quantiles: [0,5,10,...,95,100]`
- Enable correlation preservation: `correlations: {method: gaussian_copula}`
- Check for data drift (profile may be outdated)

**Memory issues**
- Enable streaming: `--streaming --chunk-size 100`
- Reduce parallel workers: `--parallel 2`
- Use JSONL output instead of JSON
- Enable checkpointing: `--checkpoint-interval 1000`

**Privacy violations**
- Increase k: `k_anonymity: {k: 10}`
- Generalize quasi-identifiers more aggressively
- Enable differential privacy: `differential_privacy: {enabled: true, epsilon: 1.0}`
- Reduce rare category threshold: `rare_category_threshold: 0.05`

---

## License & compliance

- **MIT** (default) unless your org requires Apache-2.0.
- No outbound data flows beyond your local LLM endpoint.
- Optional privacy guarantees (k-anonymity, DP) for regulated industries.
- Audit logs available for compliance tracking (GDPR, CCPA).

---

## Appendix A — JSONPath conventions

- Use `$.root.arr[]` for array elements.
- All field paths in `ruleset.yaml` are **absolute JSONPath** keys.
- Array indexing: `$.orders[0].id` (specific) vs `$.orders[].id` (all elements).
- Wildcards: `$.*.amount` (all top-level keys), `$..amount` (recursive descent).

---

## Appendix B — Statistical test reference

| Test | Type | Use case | Interpretation |
|------|------|----------|----------------|
| Kolmogorov-Smirnov | Continuous | Compare distributions | p > 0.05: distributions match |
| Anderson-Darling | Continuous | Compare distributions (tail-sensitive) | Critical value comparison |
| χ² test | Categorical | Compare frequencies | p > 0.05: frequencies match |
| G-test | Categorical | Compare frequencies (likelihood ratio) | p > 0.05: frequencies match |
| PSI | Both | Measure distribution shift | PSI < 0.1: stable, 0.1–0.25: moderate shift, >0.25: significant shift |
| Wasserstein | Continuous | Earth mover's distance | Lower is better, scale-dependent |
| Cramér's V | Categorical | Association strength | 0: no association, 1: perfect association |

---

## Appendix C — LLM prompt templates

### Profile map prompt (complete)
```
SYSTEM:
You are a data profiler. Analyze the provided JSON data and return structured insights.
Focus on accuracy and provide confidence scores for uncertain inferences.

USER:
Analyze this JSON data and return ONLY valid JSON matching the schema below.

OUTPUT_SCHEMA:
{
  "field_summaries": {
    "<jsonpath>": {
      "types": [{"name": "string|number|boolean|datetime|object|array|null", "confidence": 0.0-1.0}],
      "required_rate": 0.0-1.0,
      "enum_candidates": [{"value": "...", "count": 0}],
      "regex_candidates": [{"pattern": "...", "support": 0, "generality": 0.0-1.0}],
      "length": {"min": 0, "p25": 0, "p50": 0, "p75": 0, "p95": 0, "max": 0},
      "numeric": {"min": 0.0, "p5": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p95": 0.0, "max": 0.0},
      "datetime": {"min": "ISO8601", "max": "ISO8601", "by_weekday": [7 ints], "by_hour": [24 ints]},
      "pii_likelihood": 0.0-1.0,
      "anomalies": [{"value": "...", "reason": "..."}]
    }
  },
  "keys": {
    "pk_candidates": [{"paths": ["$.field"], "uniqueness": 0.0-1.0, "confidence": 0.0-1.0}],
    "fk_candidates": [{"parent": "$.field", "child": "$.field", "coverage": 0.0-1.0, "confidence": 0.0-1.0}]
  },
  "functional_dependencies": [
    {"lhs": ["$.field"], "rhs": ["$.field"], "support": 0.0-1.0, "confidence": 0.0-1.0}
  ],
  "temporal_patterns": [
    {"field": "$.field", "trend": "increasing|decreasing|stable", "seasonality": "daily|weekly|monthly|none"}
  ]
}

DATA_CHUNK:
{{data_chunk_json}}

CHUNK_ID: {{chunk_id}}
SEED: {{seed}}
```

### Generate text prompt (complete)
```
SYSTEM:
Generate realistic text values for a specific field in a dataset.
Follow ALL constraints exactly. Return ONLY a JSON array of strings.

USER:
FIELD: {{field_path}}
SEMANTIC_TYPE: {{semantic_type}}

CONSTRAINTS:
- regex: {{regex_pattern}}
- min_length: {{min_length}}
- max_length: {{max_length}}
- style: {{style_description}}

EXAMPLES (real data samples):
{{#each examples}}
- "{{this}}"
{{/each}}

GENERATION_PARAMS:
- start_index: {{start_index}}
- count: {{count}}
- seed: {{seed}}

Requirements:
1. Return a JSON array with EXACTLY {{count}} strings
2. Each string MUST match the regex: {{regex_pattern}}
3. Each string MUST be between {{min_length}} and {{max_length}} characters
4. Maintain stylistic consistency with the examples
5. Ensure diversity: avoid repetitive patterns

Return ONLY the JSON array, no other text:
["value1", "value2", ...]
```

---

*Enhanced for production use with **GLM-4.5-Air** on **llama.cpp**. Supports advanced statistical methods, privacy guarantees, and extensibility via plugins.*
