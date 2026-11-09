"""Analytical Benchmark Configuration.

Centralized hyperparameters, dataset selection, and kernel backend targets
for the DNN / GNN / Sheaf Neural Network comparative benchmark.
"""

# ---------------------------------------------------------------------------
# Dataset selection
# ---------------------------------------------------------------------------
DATASET = "cora"  # "cora" | "texas" | "chameleon" | "squirrel"
DATA_DIR = "data"
DOWNLOAD_URLS = {
    "cora": {
        "content": "https://raw.githubusercontent.com/kimfyoon/gat/master/data/cora/cora.content",
        "cites": "https://raw.githubusercontent.com/kimfyoon/gat/master/data/cora/cora.cites",
    },
}

# ---------------------------------------------------------------------------
# Model hyperparameters (shared across DNN, GNN, SNN)
# ---------------------------------------------------------------------------
HIDDEN_DIM = 64
NUM_LAYERS = 3
DROPOUT = 0.5
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
EPOCHS = 200
PATIENCE = 30

# ---------------------------------------------------------------------------
# Sheaf Neural Network specific
# ---------------------------------------------------------------------------
SHEAF_STALK_DIM = 4
SHEAF_TAU = 0.15
SHEAF_NUM_DIFFUSION_LAYERS = 2

# ---------------------------------------------------------------------------
# Kernel backend selection
# ---------------------------------------------------------------------------
KERNEL_BACKEND = "pytorch"  # "pytorch" | "rust" | "cuda" | "mojo"
RUST_BENCH_BINARY = r"D:\kernal\sheaf_conveyor_rust\target\release\bench.exe"

# ---------------------------------------------------------------------------
# Train / validation / test split ratios
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.10
VAL_RATIO = 0.10
TEST_RATIO = 0.80

RANDOM_SEED = 42
