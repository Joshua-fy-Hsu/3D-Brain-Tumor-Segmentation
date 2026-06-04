import torch
import os

# --- PATH CONFIGURATION ---
# Path to the preprocessed BraTS-2021 dataset (output of
# src/preprocessing/optimizing.py). Override per machine with the
# BRATS_DATA_PATH environment variable; defaults to ./data/BraTS2021_Optimized
# relative to the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_DATA_PATH = os.environ.get(
    "BRATS_DATA_PATH",
    os.path.join(_REPO_ROOT, "data", "BraTS2021_Optimized"),
)

# List of MRI modalities used as input channels.
MODALITIES = ["t1", "t1ce", "t2", "flair"]

# 4 label classes + 1 foreground one-hot channel = 5 input channels total.
NUM_CLASSES = 4
IN_CHANNELS = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Patch size increased from 96 to 128 for larger receptive field (requires ~1.1GB VRAM at batch=2).
PATCH_SIZE = (128, 128, 128)

# --- DATA SPLIT ---
TRAIN_COUNT = 1000

# --- HYPERPARAMETERS ---
BATCH_SIZE = 2        # Reduced from 4 to fit 128^3 patches in 8GB VRAM.
ACCUM_STEPS = 16      # Effective batch = BATCH_SIZE * ACCUM_STEPS = 32.
NUM_WORKERS = 6       # i5-13500HX has 14 cores / 20 threads; 6 workers fit comfortably.
PREFETCH_FACTOR = 4   # middle ground: enough buffer to keep GPU fed, less pinned RAM than 6.
PIN_MEMORY = True
SEED = 67

# --- TRAINING SCHEDULE ---
NUM_EPOCHS = 300
LR = 1e-4
WEIGHT_DECAY = 1e-5
WARMUP_EPOCHS = 5     # Linear LR warmup before cosine annealing.

# --- EARLY STOPPING ---
EARLY_STOP_PATIENCE = 300   # raised for one-shot run: only kill clearly-dead training, don't cut off cosine decay's late fine-tuning phase
EARLY_STOP_MIN_EPOCH = 50  # don't stop before this epoch (let LR warmup + early plateau pass)

# --- TRANSFORMER REGULARIZATION ---
# Stochastic depth + a single low-dose Dropout3d at the final decoder stage
# (set via the registry kwargs decoder_dropout_inner=0.0,
# decoder_dropout_final=0.05). The four attention-side dropouts are zeroed
# because they stack redundantly on a 1k-volume training set. DECODER_DROPOUT
# is kept as a legacy default for any caller that still passes it positionally.
DROP_PATH_MAX = 0.10
ATTN_DROP     = 0.0
PROJ_DROP     = 0.0
MLP_DROP      = 0.0
TOKEN_DROP    = 0.0
DECODER_DROPOUT = 0.10
