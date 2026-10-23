# Changelog

## [0.1.0] - 2026-10-22

### Added
- `PriceTable`: prices with `source` and `checked_at`; unchecked, stale and unchecked-cache lists; sanity check; cost per call with cache, thinking, high tier by whole prompt, batch, off-peak window, per-request and per-image fees; `None` for unknown models.
- `usage.extract`: Gemini, OpenAI Chat and Responses, DeepSeek cache field, Anthropic (cache reads and writes added to input). OpenAI reasoning is not counted twice.
- `Ledger` with JSONL, CSV, SQLite, memory and callback sinks; never raises, keeps `errors`; `target()` context for per-product attribution; `timer()`; `summarize` and `read_rows`.
- `errors`: account vs service vs transient; `should_retry`.
- `tokens`: `measure_factor`, `safe_ceiling` by model tier, `cliff_check` with the provider's counter.
- `budget`: `allocate` in rounds, `cut_at_boundary` with the list of items left out, `boundary_cutter`.
- `guard.unmetered` and CLI: `prices`, `cost`, `report`, `check`.
