"""Global configuration for the FedPRISM at-risk screening experiment.

All hyper-parameters, random seeds, and paths are defined here so that every
experiment in the project is fully reproducible from a single entry point.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_SEED = 42

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# The SAPEx-D dataset (already one-hot encoded).  Distributed on reasonable
# request from its authors; place the file at this path to run the pipeline.
DATASET_FILE = DATA_DIR / "sapex_d.csv"

# --------------------------------------------------------------------------- #
# Student categorisation (per the university grading policy)
# --------------------------------------------------------------------------- #
GOOD_LOWER, GOOD_UPPER = 3.67, 4.00     # "Good"
AVG_LOWER, AVG_UPPER = 2.67, 3.67       # "Average"
POOR_LOWER, POOR_UPPER = 1.00, 2.67     # "Poor"

# --------------------------------------------------------------------------- #
# Feature selection
# --------------------------------------------------------------------------- #
TOP_ACADEMIC_FEATURES = 30
PERSONALITY_TRAITS = [
    "Openness", "Conscientiousness", "Extraversion",
    "Agreeableness", "Neuroticism",
]
TOTAL_SELECTED = TOP_ACADEMIC_FEATURES + len(PERSONALITY_TRAITS)  # 35

# --------------------------------------------------------------------------- #
# Federated learning
# --------------------------------------------------------------------------- #
N_CLIENTS = 4          # simulated IID shards, not observed departments
FED_ROUNDS = 5         # communication rounds for the federated autoencoder
# Overall client-level (epsilon, delta)-DP budgets for the AE update mechanism.
# A separate non-private setting (no added noise) is labeled explicitly.
DP_EPSILONS = [0.5, 1.0, 2.0, 5.0, 10.0]
DP_DELTA = 1e-5         # overall delta across FED_ROUNDS
DP_CLIP_NORM = 1.0      # L2 clipping norm for each whole client update
META_FOLDS = 5
BOOTSTRAP_REPS = 2000

# --------------------------------------------------------------------------- #
# Autoencoder
# --------------------------------------------------------------------------- #
AE_EPOCHS = 50
AE_LATENT_DIM = 8
AE_HIDDEN_DIM = 16
AE_LEARNING_RATE = 0.01

# --------------------------------------------------------------------------- #
# Ensemble models
# --------------------------------------------------------------------------- #
IF_ESTIMATORS = 200
RF_ESTIMATORS = 200
ANOMALY_CONTAMINATION = "auto"
TEST_SPLIT = 0.20

# --------------------------------------------------------------------------- #
# Anomaly labelling (binary at-risk task)
# --------------------------------------------------------------------------- #
# On-track (Good + Average) = normal (0); at-risk (Poor) = anomaly (1).
NORMAL_LABEL, ANOMALY_LABEL = 0, 1
