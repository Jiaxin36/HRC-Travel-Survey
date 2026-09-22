# HRC: Hybrid Reweighted Calibration

Code for *Cost-Effective Travel Survey Simulation with LLMs: Hybrid Reweighted Calibration Anchored to Human Benchmarks* by Jiaxin Lu, Sijin Wu, Jiahang Liu, Lili Yang and Qiyang Liu (manuscript).

This repository provides HRC calibration, baseline comparisons, evaluation, prompt templates and fictional example data. Experiments take pre-generated LLM responses as input.

## Install and run

Python 3.12; tested on Windows CPU. Run from the repository root in a virtual environment.

```sh
python -m pip install -r requirements.txt
python run_experiments.py --country china --mode comparison --demo --output generated/china_comparison
python run_experiments.py --country england --mode comparison --demo --output generated/england_comparison
```

| Mode | Results |
|---|---|
| `comparison` | Five-method metrics and weighting diagnostics |
| `efficiency` | Metrics across training sizes |
| `robustness` | Score-component ablation and parameter sensitivity |
| `structure` | Pairwise associations and category probabilities |

Choose a new output directory for each run. Results are saved as CSV files with a JSON record of run settings. Demo settings are reduced. For research runs, replace `--demo` with `--data-dir data/private/china` or `data/private/england`. Standard runs use 600 calibration records and five seeds; efficiency uses the configured size grid.

## Local inputs

Each dataset directory contains:

```text
calibration_pool.csv     # ID, conditions, outcomes and geographic stratum
benchmark.csv            # ID, conditions and outcomes
llm/DeepSeek.csv         # ID and coded outcomes
llm/Qwen.csv
llm/GPT4.csv
llm/GPT5.csv
```

IDs must be unique, pool/benchmark IDs disjoint, and response IDs identical to benchmark IDs. Country JSON files define field roles, profile labels, answer codes and model parameters. Use coded fields as shown in the examples. Questionnaire codes determine fixed category supports and ordered bounds.

To validate and normalize existing processed tables:

```sh
python data_utils.py --country china --pool local_inputs/china/pool.csv --benchmark local_inputs/china/benchmark.csv --llm-dir local_inputs/china/llm --output data/private/china
python data_utils.py --country england --pool local_inputs/england/pool.csv --benchmark local_inputs/england/benchmark.csv --llm-dir local_inputs/england/llm --output data/private/england
```

Preparation preserves the supplied split; establish eligibility and household separation upstream. Prompt templates document the generation procedure; use your own generation implementation and map answers to the configured codes.

## Files and data availability

`run_experiments.py` runs experiments; `hrc.py` implements calibration; `evaluation.py` provides baselines and metrics; `data_utils.py` loads configurations and validates data. `configs/` contains country and experiment settings; `prompts/` contains the two templates.

China and England survey records and respondent-linked outputs are confidential and excluded. All bundled CSVs are wholly fictional; named LLM demo files contain mock answers, not actual model responses. Demos verify execution, not empirical performance.

The code is released under the MIT License; data access remains separately restricted.
