# FedPRISM: at-risk screening experiment

Reproducible Python code for a retrospective student at-risk classification experiment. The repository contains the modeling and evaluation pipeline, dependency list, and aggregate result artifacts. It does **not** contain the SAPEx-D dataset or student-level test predictions.

## Scope, data access, and privacy

- **Dataset access:** SAPEx-D is available from its authors on reasonable request. This repository does not host the dataset, and no public dataset-repository URL is available. Place an authorized copy at `data/sapex_d.csv` to run the experiment.
- **No data bundled:** no raw SAPEx-D records or synthetic replacement data are included. Do not commit or redistribute the dataset.
- **Row-level output:** running the pipeline creates `results/test_predictions.csv` locally. It contains held-out labels and per-student predictions/scores, is excluded by `.gitignore`, and must not be committed, uploaded, or shared. The checked-in result files are aggregate tables and figures only.
- **Federation:** the four clients are reproducible, approximately equal-size random IID shards. This is a single-process simulation—not a deployed network or a partition by actual departments or institutions.
- **Differential privacy:** the primary FedPRISM run uses unnoised updates and is **not end-to-end differentially private**. The separate Gaussian/RDP sensitivity experiment covers client-level privacy for released FedAvg autoencoder parameters only. It does not cover the Isolation Forest, Random Forest, logistic stacking model, guidance outputs, or the coordinator's view of local updates. **Secure aggregation is not implemented.** Do not describe the complete pipeline as private or deployment-ready.
- **Interpretation:** the repository reports point estimates and uncertainty. Paired intervals include zero; point estimates should not be presented as proof of superiority over the Random Forest or as reliable incremental benefit from Big-Five inputs. Candidate academic-guidance prompts are for human review and are not validated interventions or causal prescriptions.

## Experiment specification

- 494 complete records, 86 encoded predictors, and outcome fields `Exact CGPA` and `Letter Grade`; the at-risk label is derived from `Exact CGPA`.
- Seed-42 stratified 80/20 split: 395 training records and 99 held-out records.
- Scaling and F-ANOVA feature selection are fitted on training data only, including inside cross-fitting folds.
- Primary FedPRISM input: 35 features—30 selected non-personality predictors plus all five Big-Five traits. The `FedPRISM without Big Five` ablation uses 30 non-personality features.
- Five-fold out-of-fold (OOF) detector scores train the logistic stacking model and calibrate component thresholds; the held-out test data are not used for either step.
- Four random IID clients are simulated. FedAvg uses five communication rounds for the autoencoder; local Random Forest and Isolation Forest scores are combined in the simulated pipeline.
- The AE-only privacy-sensitivity sweep uses overall epsilon values 0.5, 1, 2, 5, and 10, delta `1e-5`, and clipping norm `1.0`; no subsampling amplification is assumed.
- Reported 95% intervals use 2,000 class-stratified bootstrap resamples of the fixed 99-record test set. They condition on the fitted models and do not measure retraining or across-cohort variability. False-alarm rate is `FP / (FP + TN)`.

## Repository contents

```text
.
├── main.py                 # single entry point
├── experiment.py           # cross-fitting, evaluation, ablations, bootstrap, DP sweep
├── analysis_figures.py     # aggregate and descriptive figures
├── config.py               # paths, seeds, folds, and model/privacy settings
├── preprocess.py           # input loading and training-only scaling
├── feature_selection.py    # fold-local F-ANOVA feature selection
├── federated.py            # simulated FedAvg, local models, and RDP accounting
├── models.py               # NumPy autoencoder and scikit-learn model builders
├── evaluate.py             # metrics, threshold calibration, plotting helpers
├── recommendation.py       # human-reviewed, behavior-focused prompt prototype
├── requirements.txt
├── data/
│   ├── README.md            # data-access and placement instructions
│   └── .gitkeep
└── results/                 # safe aggregate tables and figures
```

## Setup and run

Use Python 3.10 or newer.

```bash
# From the repository root
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows PowerShell: use instead
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Obtain the dataset from its authors on reasonable request, then place the authorized CSV at `data/sapex_d.csv`. The input should be the released one-hot-encoded dataset with the `Exact CGPA` and `Letter Grade` outcome columns. The file is intentionally ignored by Git.

```bash
python main.py
```

The run performs the cross-fitted federated and centralized experiments, ablations, bootstrap analysis, AE-only DP sensitivity sweep, and figure generation. It may take several minutes on a standard CPU. Results are written under `results/`; the student-level `test_predictions.csv` is generated only for local analysis and must remain unshared.

## Aggregate outputs

The checked-in `results/` directory contains aggregate artifacts from a run of this pipeline:

- `metrics.json` — aggregate metrics, uncertainty, paired comparisons, client counts, selected-feature summary, and DP scope.
- `heldout_metrics.csv`, `model_comparison.csv`, `ablation_metrics.csv` — aggregate performance and ablation tables. The complete model is labeled `FedPRISM`; detector-only rows retain their detector names.
- `client_distribution.csv`, `fedavg_history.csv`, `dp_ae_sweep.csv`, `selected_features.txt` — client counts, round-level aggregate training loss, AE-only sensitivity results, and selected feature names.
- PNG files — aggregate performance, ablation, sensitivity, and descriptive figures.

A new run overwrites the aggregate outputs and creates the local row-level prediction file. Do not force-add private data or row-level outputs to Git.



## License

The source code is distributed under the MIT License; see `LICENSE`. The dataset is not included and remains subject to its own access and use conditions.
