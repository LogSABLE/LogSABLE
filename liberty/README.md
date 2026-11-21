# Liberty dataset

This folder ships a small **templates file** in the repository. The large session CSV must be downloaded separately.

## Included in the repository

| File | Description |
|------|-------------|
| `liberty_templates.csv` | EventId ↔ LabelToken mapping (11 event types from LogLLM preprocessing) |

## Download separately

| File | Description |
|------|-------------|
| `liberty_150k.csv` | Session-level log data (~78 MB; not committed to GitHub) |

Expected columns in `liberty_150k.csv`: `LineInSlice`, `LabelToken`, `Label`, `ts`, `log`.

## Setup

1. Follow the [LogLLM repository](https://github.com/guanwei49/LogLLM) to download and preprocess the Liberty logs.
2. Place the processed CSV as `liberty/liberty_150k.csv`.
3. Keep `liberty_templates.csv` in this folder (already provided).

## Config

```yaml
data:
  dataset: "Liberty"
  data_dir: "liberty"
  liberty_filename: "liberty_150k.csv"
  liberty_templates_filename: "liberty_templates.csv"
  liberty_bucket_sec: 100
```

For LogAnomaly on Liberty, the pipeline uses `LabelToken` events via `liberty_templates.csv` (same pattern as BGL/HDFS template files).
