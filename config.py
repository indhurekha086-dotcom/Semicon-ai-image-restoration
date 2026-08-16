import os

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Dataset paths
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

# Image settings
IMG_SIZE = 256
IN_CHANNELS = 3
OUT_CHANNELS = 3

# Training settings
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
NUM_WORKERS = 2

# Device
DEVICE = "cuda"
