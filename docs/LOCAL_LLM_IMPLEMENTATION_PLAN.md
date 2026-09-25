# Local specialist analyst: implementation plan

Status: proposed implementation, 2026-09-23. No local model run or training result is implied. The current change documents the work; all implementation stages below remain pending. Architecture and contracts: [system design](ARCHITECTURE.md).

## 1. Fixed starting experiment

User-selected baseline: **Qwen3.5-4B, 4-bit quantisation, 4,096 total tokens**, on MacBook Air M1 / 8 GB unified memory / 256 GB storage. Start at this profile before considering Qwen3.5-2B. Do not silently start at 2,048 tokens or substitute 2B. Actual runtime support and memory fit must be measured.

| Setting | Initial decision |
| --- | --- |
| Model | Post-trained `Qwen/Qwen3.5-4B`; verified compatible GGUF conversion |
| Quantisation | Q4_K_M candidate; exact artifact pinned and hashed |
| Backend | Native llama.cpp with Metal, compatible pinned build |
| Context | 4,096 per sequence, prompt plus generation |
| Generation reserve | 768 tokens; rendered input at most 3,328 |
| Mode | Text only, verified direct-answer mode, no image projector |
| Concurrency | One model, one active request; bounded busy rejection |
| Sampling | Initial temperature 0.2 and seed 42 where supported; record all effective settings and repeat for variability |
| Initial request deadline | 120 seconds; proposed usability target, not a measured capability |
| Model endpoint | Loopback `127.0.0.1:8081`; application remains on port 8000 |
| Storage | New experiment directory per run; check actual artifact size and retain at least 20 GiB free as an initial operating target |
| Forecast influence | Disabled; analyse existing numerical forecasts |

The direct-answer profile is an experiment, not the vendor's full benchmark configuration. If unsupported, report that failure before choosing a different profile. The 4,096 limit remains the first comparison condition for 2B and alternative models.

## 2. Delivery sequence and dependencies

| Stage | Deliverable | Depends on | Completion evidence |
| --- | --- | --- | --- |
| 0 | Hardware/model feasibility experiment | None | Reproducible 4B/4,096 report or documented failure |
| 1 | Provider adapter, configuration and contracts | 0 | Offline contract tests and pinned-server smoke test |
| 2 | Cutoff-aware evidence store and retrieval | 1 | Retrieval/provenance tests and fixtures |
| 3 | Read-only analyst and numerical tool integration | 1, 2 | Audited answer through CLI/API; forecast invariance |
| 4 | Quality/resource benchmark and model decision | 0–3 | Frozen evaluation report and explicit 4B/2B decision |
| 5 | Review workflow and analyst dataset builder | 3, 4 | Reviewed, reproducible, leakage-checked dataset |
| 6 | Adapter training and exported-model evaluation | 5 plus hardware feasibility | Candidate report or explicit no-training decision |
| 7 | Versioned release, rollback and operational checks | 4; 6 if using adapter | Reproducible local installation and rollback exercise |
| 8 | Optional forecast-affecting experiment | 7 plus sufficient evidence | Existing protected numerical gates and prospective audit |

Stages 0–4 produce a usable base-model analyst. Stages 5–7 add controlled specialisation and releases. Stage 8 is optional and is not a prerequisite for an analyst. No forecast benefit is assumed from an explanation-quality improvement.

## 3. Stage 0 — measure 4B at 4,096 first

### Work

1. Capture macOS version, CPU architecture, installed RAM, free storage, power mode and ordinary background applications. Do not infer available RAM from installed RAM. Record idle memory pressure and swap usage.
2. Select and record a trustworthy GGUF conversion, source revision/license, file hash and tokenizer/template provenance. Avoid downloading several full-precision checkpoints or multiple caches. Verify the quantisation metadata.
3. Install or build a pinned native llama.cpp version with Metal. Record build options and verify the selected model architecture loads. Confirm one slot and effective 4,096 context from runtime metadata/logs.
4. Verify chat templating, direct-answer mode, token counting, JSON-schema response behaviour and cancellation before integration. The proposed configuration must be checked against that build's documented flags.
5. Run six short smoke prompts: factual extraction, numeric explanation, unknown evidence, malformed input, quoted malicious instructions and a near-limit prompt. Include a synthetic 22-driver compact forecast so small four-driver examples do not conceal context pressure.
6. Run a 50-case development benchmark, balanced across five task groups, with one warm-up and two timed repetitions in varied order. Measure a cold load separately. Include a 20-request continuous workload to expose sustained slowdowns, then repeat a representative subset with the forecast service active.
7. Write raw responses, settings, timings, memory/swap observations, validation outcomes and a readable report. On failure, retain the exact profile and cause. Adjust only one variable per subsequent experiment.

### Proposed resource gates

These are preregistered engineering targets to test, not product promises or vendor requirements:

- Zero crashes, out-of-memory failures or server-side silent truncation on the completed development set.
- Stop the stress run if memory pressure remains critical/red for 30 seconds or the machine becomes unresponsive. Record a resource failure; do not attempt to complete the run through heavy swapping.
- Record swap relative to the idle baseline. More than 1 GiB growth or continued growth across the final ten stress requests triggers resource review and the 2B experiment. Existing swap alone is not a failure.
- Target warm p95 first-token latency at most 15 seconds and p95 end-to-end time at most 90 seconds for the fixed answer-length workload; every request has the 120-second deadline. Include actual output-token lengths so a shorter answer cannot masquerade as higher throughput.
- Measure loading separately; do not hide cold starts in warm results or claim p95 from a handful of examples.

The first deliverable is `artifacts/local-llm/<run-id>/report.md` with `manifest.json`, `requests.jsonl`, `responses.jsonl`, `metrics.json` and `hardware.json`. These are future output paths, not current artifacts.

### Decision

Keep 4B if usable and correct enough for the pilot. If resource gates fail, run **2B at 4,096** with identical tasks and output limits. Retain both reports. If 2B fails too, report the limitation and separately test 2,048 or a narrower task. If quality fails while resources pass, compare alternative 4B-class checkpoints before attributing the problem to size. None of these switches is an automatic per-request fallback.

## 4. Stage 1 — provider boundary and configuration

### Proposed files

- `src/f1_forecast/llm_provider.py`: protocols, capability negotiation, result/error types.
- `src/f1_forecast/local_llm.py`: llama.cpp HTTP adapter using existing `httpx`.
- `src/f1_forecast/analysis_models.py`: strict task, evidence, number, answer and review schemas.
- `src/f1_forecast/analysis_config.py`: validated configuration, immutable model identity and budget limits.
- `tests/test_local_llm.py`: transport and parsing contracts using fake HTTP responses.
- `scripts/benchmark_local_llm.py`: opt-in real-server benchmark; never part of ordinary offline tests.

Keep existing `LLM()` semantics and remote research available. Refactor its internals only with compatibility tests. Do not send local requests through hosted fine-tuning or web-search code. Keep installation of a local model runtime independent from the Python package's hosted `llm` extra.

Proposed configuration names, **not supported yet**:

```text
F1_ANALYSIS_PROVIDER=llama_cpp
F1_ANALYSIS_BASE_URL=http://127.0.0.1:8081
F1_ANALYSIS_MODEL_MANIFEST=artifacts/local-llm/releases/<release-id>/manifest.json
F1_ANALYSIS_CONTEXT_TOKENS=4096
F1_ANALYSIS_MAX_OUTPUT_TOKENS=768
F1_ANALYSIS_TIMEOUT_SECONDS=120
F1_ANALYSIS_DATABASE=data/analysis.sqlite3
```

The manifest fixes quantisation, template, sampling and reasoning mode. Existing `F1_LLM_MODEL` and `F1_OUTCOME_MODEL` retain their existing meaning. Validate the configured backend's actual model identity against the manifest before serving analysis.

Implement exact rendered-token counting, schema-constrained generation when supported, local Pydantic validation and structured errors. Count schema/tool overhead; reject overflow before generation. A repair attempt must use a freshly budgeted prompt and the remaining overall deadline. Do not execute tools from a malformed output.

### Acceptance

Offline tests cover valid/invalid JSON, extra fields, refusal, empty content, length termination, mismatched model identity, unsupported schema, timeout, disconnect, cancellation, full slot, overflow and one bounded repair. A separate real-server smoke confirms these mappings on the pinned runtime. Existing hosted LLM integration tests remain green. No model downloads or network calls occur during the unit suite.

## 5. Stage 2 — evidence ingestion and retrieval

### Proposed files

- `src/f1_forecast/analysis_store.py`: analysis database, migrations, audit operations.
- `src/f1_forecast/evidence_store.py`: immutable source ingestion, chunks, FTS retrieval and cutoff policy.
- `src/f1_forecast/context_budget.py`: task-specific compact projection and token-aware packing.
- `tests/test_evidence_store.py`, `tests/test_context_budget.py`.

Implement the tables and source-time policy in [architecture sections 5–6](ARCHITECTURE.md). Start from imported text/JSON and known source metadata. Use chunks of approximately 200–350 model tokens as a development setting, preserving offsets and paragraph boundaries; tune retrieval on development data. Retrieve up to six passages initially, then pack only those fitting the evidence budget. Exact SQL facts do not require embeddings.

Keep source documents immutable: a corrected or updated page creates a new version and hash. Parameterise SQL/FTS queries and normalise identifiers; never let the LLM construct executable SQL. Store query/version/chunk selection for replay. Surface missing publication times and exclusion reasons to the analyst.

### Acceptance

Fixtures cover a document arriving after the cutoff, an updated page, unknown publication time, unsupported historical availability, future target results, duplicate passages, contradictory sources, UTF-8 quote offsets, prompt injection in a passage and no eligible evidence. Boundary tests prove the rendered request and output allowance fit exactly 4,096. A full-field snapshot must either compact correctly with explicit omitted fields or return an honest budget error. Rebuilding FTS must reproduce the same eligible document set.

## 6. Stage 3 — analysis service, tools and user interface

### Proposed files and integration points

- `src/f1_forecast/analysis_service.py`: orchestrates retrieval, tools, generation, validation and audit.
- `src/f1_forecast/analysis_tools.py`: typed allowlisted readers and deterministic numerical facts.
- `src/f1_forecast/cli.py`: add separate analysis/import/review commands.
- `src/f1_forecast/api.py`: add analysis routes without changing prediction defaults.
- `tests/test_analysis_service.py`, `tests/test_analysis_api.py`.

Initial task types: `explain_forecast`, `compare_scenarios`, `extract_evidence` and `assess_input_quality`. The controller selects tools deterministically from task type. Tool names are application interfaces, not promises that all underlying data are available; return explicit unsupported/missing-data results.

Execution sequence:

1. Load a saved forecast/snapshot through the existing store and record its hash.
2. Derive the cutoff/mode and retrieve eligible evidence.
3. Compute compact numerical facts from stored outputs; reference their exact IDs and units.
4. Render/count the prompt, prune lower-priority evidence, and generate one structured answer.
5. Validate schema, IDs, quote locations and number references; render verified numbers in the final answer.
6. Save result, source selection, raw response and timings. Return unknowns/limitations alongside the answer.

Answer verification cannot be delegated solely to another LLM. Deterministic checks protect structure/numbers; reviewed evaluation checks whether claims are actually supported. A validated response is not a guarantee that every inference is correct.

Planned CLI examples below are **interface targets, not runnable commands today**:

```sh
f1-forecast analysis-import documents.json
f1-forecast analyze --forecast-id FORECAST_ID --task explain_forecast \
  --question 'Which supplied factors explain this forecast?' --output analysis.json
f1-forecast analysis-review --analysis-id ANALYSIS_ID --review review.json
```

Expose the routes described in the architecture with strict request limits and one generation slot across requests. Do not change the current `predict --llm` command into this workflow. The local analyst requires no hosted API credential. An unavailable model returns an explicit error while ordinary numerical prediction remains available.

### Acceptance

End-to-end fixtures prove forecast values and numerical learner state are unchanged before/after analysis. Test cross-request isolation, empty evidence, incorrect driver IDs, fabricated sources/numbers, timed-out calls and recovery. An analysis review must not trigger `Store.feedback()` or any weight update. Run CLI/API tests without a real model and one opt-in local-server workflow with the real artifact.

## 7. Stage 4 — evaluate quality and select a model

Build a versioned task set and a manual scoring guide. Start with 50 development cases; before declaring a release, prepare a separate 100-case test set with 20 cases in each category:

1. Evidence extraction with exact support.
2. Numerical forecast/scenario explanations.
3. Missing/contradictory information and abstention.
4. Cutoff, provenance and document-instruction attacks.
5. Context-boundary, malformed-output and full-field cases.

Categories describe the primary challenge; task types may overlap. Use verified F1 examples plus clearly labelled synthetic robustness fixtures. Keep synthetic robustness results separate from real F1 quality. Keep weekends, source versions and paraphrases together across splits. If the available corpus cannot support these counts without leakage, report the gap and remain an experimental pilot.

For each model record base revision, quantisation, runtime, context, prompt/template, effective sampling, evidence ordering and output limit. Candidate-specific templates are necessary; keep the semantic task and supplied evidence fixed. Vendor benchmark scores are background information, not our selection metric.

### Initial release gates

| Metric | Proposed gate |
| --- | --- |
| Schema validity | At least 95% on the first pass; report repair outcomes separately |
| Validity of accepted answers | 100% valid schema, source IDs and referenced numbers |
| Critical failures | Zero accepted future-evidence leaks, executed document instructions, fabricated numerical facts or fabricated source IDs |
| Reviewed claim support | At least 95% of substantive claims supported by supplied evidence/tools |
| Required fact coverage | At least 85% of required gold facts on answerable cases, preventing empty answers from passing |
| Appropriate abstention | At least 90% on cases labelled unanswerable |
| Resource behaviour | Stage 0 stability and latency targets on the complete service |
| Forecast isolation | All invariance tests pass |

Report denominators, task/session breakdowns, confidence intervals where meaningful and all failures. Small test sets do not prove absence of rare failures. Freeze thresholds before model selection; if a target changes, version the evaluation protocol and explain why. A model that abstains on everything fails coverage.

Compare 4B against 2B at 4,096 when fallback is needed or when testing the resource/quality tradeoff. Compare alternative 4B candidates on the development set, choose there, then run the chosen candidate on the held-out test. Do not repeatedly tune against the test; retire it into development and create a fresh later holdout if that happens.

Artifacts: `eval_manifest.json`, `per_case.jsonl`, `summary.json`, `report.md`, and a readable HTML report if useful. Include both first-pass and final results, cold/warm timings and original responses. Prospective forecast accuracy is not a metric of this analysis-only benchmark.

## 8. Stage 5 — reviewed data and growing specialisation

Add `src/f1_forecast/analysis_dataset.py`, dataset tests and an explicit `export-analysis-training` command. Keep it separate from `export-training`, which currently exports outcome ordering.

Store each review with reviewer identity, time, disposition and correction lineage. A gold analyst example contains question/task, original evidence, exact tool results, desired structured answer, grouping keys, review status and hashes. Confirm sources and facts before inclusion; exclude failed, unreviewed or unsupported self-generated responses. Include representative abstentions, conflicting evidence and correction examples.

Export `train.jsonl`, `validation.jsonl`, `test.jsonl` plus a manifest. Keep whole weekends/source groups together, order by availability, purge delayed labels/reviews and deduplicate before splitting. Test data remain out of training jobs and prompt tuning. Respect source licenses and record allowed usage before building a redistributable dataset. Export content only when permitted; local access does not establish redistribution rights.

Pilot planning target: 300–1,000 carefully reviewed examples across the supported tasks, expanding based on failure coverage. This is a collection target, not a guarantee that fine-tuning will help. First compare retrieval/prompt improvements against adapter training. Standardise schemas across models so the same reviewed corpus can train 4B or 2B independently.

### Acceptance

Repeated export from the same source snapshot produces identical hashes/splits. Tests reject tampering, shared weekend/document groups, unverified labels, future input evidence, delayed training labels crossing the validation boundary and synthetic/real mixing. Reviews and outcome feedback remain separate. Dataset creation never launches training or uploads data.

## 9. Stage 6 — adapter training feasibility and experiment

### Hardware branches

- **Mac-only:** stop the inference server, verify exact MLX architecture/trainer support, then run a bounded small-model feasibility probe. A 0.8B or 2B probe is a separately labelled training experiment; it does not replace the initial 4B inference evaluation. Start batch size 1 and short training sequences, measure memory and one checkpoint save/reload. Stop on pressure; do not promise 4B training on 8 GB.
- **Larger training machine:** optional explicit choice when ready. Train the selected 4B/2B base with a supported LoRA recipe, then bring back a compatible inference export. The present plan does not authorise paid compute or uploading the user's data.

Current Qwen3.5 training guidance cautions against four-bit QLoRA. Prefer BF16 LoRA where supported and sufficiently resourced; treat any other recipe as a separately evaluated experiment. CUDA VRAM estimates do not translate directly into Mac unified memory. Keep MLX/accelerator dependencies in a dedicated training environment rather than the core forecasting environment.

### Proposed experiment

1. Pin base revision, tokenizer, template, dataset and trainer versions; record hashes.
2. Verify one forward/backward update and save/reload without data corruption.
3. Start with language-only supervised targets and frozen base weights; choose supported target modules from the actual architecture, not a copied module list from another model.
4. Initial development search: LoRA rank 8 or 16, learning rate 1e-5 or 5e-5, up to three epochs with validation-based early stopping. These are trial values; predeclare the chosen search and resource cap before a full run. Record effective batch size, accumulation, precision, sequence length and seed.
5. Train loss on intended assistant targets; preserve the inference chat template and include reviewed tool results in context. Do not train on padding or accidentally include held-out targets in prompts.
6. Check both specialist tasks and a fixed small general-instruction regression set. Select on validation, then evaluate the chosen candidate on the untouched test set.
7. Merge/export only through a supported path. Validate unquantised candidate versus base, then compare final 4-bit export against the candidate and base on the same tasks.
8. Run the exported release on the Mac at **4,096**, even if training used shorter sequences. Record context-boundary regressions.

### Promotion rule

The adapter must meet every Stage 4 gate, introduce no critical regression, and demonstrate a predeclared improvement on the targeted task metric against base plus identical retrieval. Use paired case/group comparisons and uncertainty; insufficient evidence retains the base. More training loss reduction alone is not a release criterion. Run additional seeds when the gain is small or variable. Record a no-improvement result rather than automatically publishing a new version.

Save base reference, adapter, configuration, learning curves, dataset manifest, runtime export hash and evaluation reports. A 4B-to-2B change trains a new adapter. Optional distillation uses reviewed teacher outputs and independent evaluation; teacher-generated text is not automatically ground truth.

## 10. Stage 7 — release and operation

Implement `src/f1_forecast/analysis_registry.py` and tests for manifests, compatibility and atomic selection. Preserve the previous good artifact; stop/start sequentially to avoid loading two models into 8 GB. Verify identity, health, schema and a known evidence task before selecting a release. Exercise a failed startup and successful rollback.

Provide a runbook covering installation, exact model acquisition, pinned server invocation, effective 4,096 verification, start/stop, expected ports, cancellation, resource troubleshooting, dataset backup/restore, FTS rebuild and release rollback. Replace illustrative interfaces with tested commands only after implementation. Avoid baking model weights into Git; add `*.gguf` and relevant cache paths to ignore rules when downloads are introduced. Existing `artifacts/` is already ignored.

Cache only after correct versioned invalidation is tested. Start with no automatic boot service and no scheduled retraining. Every training run and model selection is explicit. A changed corpus can be indexed independently without changing the active model release.

### Acceptance

A documented setup reproduces the selected model on the target Mac. Startup checks reject wrong hashes/template/context. Offline analysis works after installation. Backups restore reviewed examples and evidence relationships. Broken candidates roll back without changing numerical models or losing analysis history. The release report includes remaining limitations and a clear distinction between tested behaviour and future scope.

## 11. Stage 8 — optional numerical influence

Only after a useful analyst exists, evaluate bounded extracted features or scenario proposals as separate forecast candidates. Do not enable them by simply pointing the existing `--llm` switch at the new server.

Predeclare the candidate mapping and retain the numerical-only baseline. Use the existing chronological windows, complete seed panels, paired-weekend uncertainty and session-specific gates from [guarded improvements](GUARDED_IMPROVEMENTS.md) and [session mixtures](SESSION_MIXTURES.md). Protect position MAE, Brier/log loss, ranking accuracy and interval quality. Keep synthetic fixtures separate and record input availability. Reject any candidate with insufficient support or protected regressions.

Historical LLM result recall remains a limitation. Freeze timestamped live predictions before outcomes, score after verified feedback and retain a prospective audit. Analyst citation accuracy and attractive explanations cannot substitute for this evidence.

## 12. Verification and completion checklist

During implementation, run focused offline tests for each stage, then the relevant existing integration/training/forecast tests. Before a release run the complete offline suite and repository lint/format checks. Hardware benchmarks are explicit opt-in checks and must report the actual Mac and model identity. Training smoke tests are separate and never run automatically in the standard test suite.

- [ ] Stage 0: 4B, 4-bit, 4,096 measured first; effective settings and failures preserved.
- [ ] Stage 1: Provider contracts and exact budgets tested; hosted workflow preserved.
- [ ] Stage 2: Immutable evidence, cutoff filtering and retrieval tested.
- [ ] Stage 3: Read-only analyst works through CLI/API; forecasts unchanged.
- [ ] Stage 4: Quality and resource gates measured; explicit model decision recorded.
- [ ] Stage 5: Reviewed analyst dataset exports reproducibly with safe splits.
- [ ] Stage 6: Training feasibility recorded; adapter promoted only with evidence, or base retained.
- [ ] Stage 7: Local release and rollback verified; operational guide runnable.
- [ ] Stage 8, optional: Forecast influence separately gated and prospectively tracked.

For this documentation-only change, check relative links, document consistency and whitespace. Do not report runtime/model tests as passed until implementation and experiments have actually run.
