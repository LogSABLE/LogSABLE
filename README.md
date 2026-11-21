# LogSABLE

**Semantic Augmentation of Backbone Log Detectors with Executable Evidence**

LogSABLE is a plug-in framework that enhances existing log anomaly detectors (DeepLog, LogAnomaly, LogBERT, NeuralLog, LogRobust) with semantic evidence from local patterns, LLM reasoning, and cluster-derived rules—without replacing the backbone training pipeline.

## Abstract

Log anomaly detection is critical for reliability monitoring, but detectors that operate mainly on parsed EventId sequences can miss semantic information carried by log text. This loss of operational semantics limits their ability to explain why a session is anomalous, even when the final anomaly label is correct. The gap matters in practice because operators need to distinguish urgent failures from benign deviations and trace suspicious sessions back to concrete operational behavior.

LLMs can recover richer log semantics, but using them as standalone detectors or repeatedly generating free-form explanations can be costly and difficult to reuse. We present **LogSABLE**, a plug-in framework that enhances existing backbone detectors without replacing their original pipelines. LogSABLE combines backbone predictions with reusable semantic evidence from local event patterns, LLM-based semantic reasoning, and cluster-derived executable rules to produce both anomaly decisions and evidence-based explanations.

We evaluate LogSABLE on **HDFS**, **BGL**, and **Liberty** using DeepLog, LogAnomaly, LogBERT, and NeuralLog as backbones. LogSABLE improves every non-perfect baseline, preserves the already perfect case, recovers 98.8% of backbone false negatives, and achieves high explanation coverage with explanations preferred over direct LLM explanations in a human study. These gains require only modest and stable inference-time overhead.

## Research questions

### RQ1 — How effective is LogSABLE for anomaly detection?

![RQ1: baseline vs LogSABLE detection performance (P, R, F1, ΔF1)](docs/images/rq1.png)

### RQ2 — How efficient is LogSABLE in runtime and LLM usage?

![RQ2: training and inference time — baseline vs LogSABLE](docs/images/rq2.png)

### RQ3 — How useful are LogSABLE explanations?

![RQ3: human evaluation of explanation quality](docs/images/rq3.png)

### RQ4 — How do different settings and training-data percentages affect LogSABLE performance?

![RQ4: component ablation](docs/images/rq4_1.png)

![RQ4: F1 under different training-set percentages](docs/images/rq4_2.png)

## Revision Results

### n-gram length sensitivity

| Dataset | Backbone | n=1 | n=2 | n=3 | n=4 | n=5 |
|---------|----------|-----|-----|-----|-----|-----|
| HDFS | DeepLog | 0.91 | 0.91 | 0.91 | 0.93 | 0.93 |
| HDFS | NeuralLog | 0.91 | 0.91 | 0.91 | 0.93 | 0.93 |
| HDFS | LogAnomaly | 0.92 | 0.92 | 0.92 | 0.93 | 0.93 |
| HDFS | LogBERT | 0.90 | 0.97 | 0.97 | 0.98 | 0.98 |
| BGL | DeepLog | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| BGL | NeuralLog | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| BGL | LogAnomaly | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| BGL | LogBERT | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Liberty | DeepLog | 0.94 | 0.94 | 0.94 | 0.94 | 0.94 |
| Liberty | NeuralLog | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 |
| Liberty | LogAnomaly | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Liberty | LogBERT | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 |

### LLM-based detectors

| Detector | Setting | HDFS P | HDFS R | HDFS F1 | BGL P | BGL R | BGL F1 | Liberty P | Liberty R | Liberty F1 |
|----------|---------|--------|--------|---------|-------|-------|--------|-----------|-----------|------------|
| LogFormer | Base | 0.93 | 0.73 | 0.82 | 1.00 | 0.95 | 0.97 | 0.63 | 0.60 | 0.61 |
| LogFormer | + LogSABLE | 0.93 | 1.00 | 0.96 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 | 0.99 |
| LogFiT | Base | 0.28 | 0.32 | 0.30 | 0.89 | 0.34 | 0.50 | 0.59 | 0.60 | 0.59 |
| LogFiT | + LogSABLE | 0.93 | 0.96 | 0.94 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 | 0.99 |
| LogLLM | Base | 0.92 | 0.99 | 0.96 | 0.95 | 0.88 | 0.92 | 1.00 | 0.60 | 0.75 |
| LogLLM | + LogSABLE | 0.99 | 0.99 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 | 0.99 |

### Prompting-based LLM methods

| Method | HDFS P | HDFS R | HDFS F1 | BGL P | BGL R | BGL F1 | Liberty P | Liberty R | Liberty F1 |
|--------|--------|--------|---------|-------|-------|--------|-----------|-----------|------------|
| LogPrompt | 0.06 | 0.72 | 0.10 | 0.68 | 0.97 | 0.80 | 0.21 | 1.00 | 0.35 |
| LogRules | 0.07 | 0.55 | 0.13 | 0.97 | 0.93 | 0.95 | 0.38 | 0.94 | 0.55 |
| LogSABLE LLM only | 0.37 | 0.22 | 0.28 | 0.82 | 0.95 | 0.89 | 0.46 | 0.71 | 0.56 |

### Fewer anomaly labels and unseen failure types

| Dataset | Budget | F1 range | Unseen recall DeepLog (%) | Unseen recall NeuralLog (%) | Unseen recall LogAnomaly (%) | Unseen recall LogBERT (%) |
|---------|--------|----------|---------------------------|-----------------------------|------------------------------|---------------------------|
| HDFS | 10% | 0.909–0.964 | 100 | 100 | 97.3 | 89.2 |
| HDFS | 25% | 0.909–0.946 | 100 | 100 | 95.2 | 90.5 |
| HDFS | 50% | 0.909–0.961 | 100 | 100 | 91.7 | 83.3 |
| BGL | 10% | 0.999–1.000 | 91.7 | 100 | 91.7 | 91.7 |
| BGL | 25% | 0.999–1.000 | 100 | 100 | 100 | 100 |
| BGL | 50% | 0.999–1.000 | 100 | 100 | 100 | 100 |
| Liberty | 10% | 0.944–1.000 | 0 | 0 | 100 | 0 |
| Liberty | 25% | 0.944–1.000 | 0 | 0 | 100 | 0 |
| Liberty | 50% | 0.944–1.000 | 0 | 0 | 100 | 0 |

## Datasets

The full datasets used in this work are not included in this repository because they occupy several gigabytes and exceed GitHub's storage limits.

Please download the datasets from their original sources and place them in the appropriate directories before running the experiments.

| Dataset | Source | Local directory |
|---------|--------|-----------------|
| HDFS | [LogHub – HDFS Dataset](https://github.com/logpai/loghub/tree/master/HDFS) | `hdfs_data/` |
| BGL | [LogHub – BGL Dataset](https://github.com/logpai/loghub/tree/master/BGL) | `bgl/` |
| Liberty | [LogLLM repository](https://github.com/guanwei49/LogLLM) — see [`liberty/README.md`](liberty/README.md) | `liberty/` *(templates included; add `liberty_150k.csv` locally)* |

### Notes

- This repository contains only the code, configuration files, and small metadata files required to reproduce the experiments.
- Large datasets were intentionally excluded to keep the repository lightweight and within GitHub's file size limits. Empty dataset folders (e.g. `liberty/`) contain a `README.md` with download and placement instructions.
- Please refer to the original dataset licenses and citation requirements when using these datasets. The LogHub datasets are freely available for research use and should be cited according to the [LogHub paper](https://github.com/logpai/loghub).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"   
```

## Run

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python src/main.py
```

Edit [`configs/default.yaml`](configs/default.yaml) for dataset, backbone model, clustering, and LLM settings.

## Repository layout

```
LogSABLE/
├── configs/default.yaml
├── src/main.py
├── src/logsable/
├── docs/images/
├── Results/
├── hdfs_data/          # HDFS files (download separately; see Datasets)
├── bgl/                # BGL files (download separately; see Datasets)
└── liberty/            # liberty_templates.csv included; add liberty_150k.csv locally
```

## Citation

If you use this artifact, please cite the accompanying paper (bibtex to be added).
