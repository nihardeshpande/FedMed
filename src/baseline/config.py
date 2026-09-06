"""
Shared constants for the baseline pipeline. Centralizing these here means
changing a patient count or path only requires editing one file, instead of
hunting through train.py, brats_dataset.py, and unet_model.py separately.
"""

# Dataset
NUM_MODALITIES = 4          # t1, t1ce, t2, flair
NUM_CLASSES = 4              # background, necrotic core, edema, enhancing tumor (remapped 4->3)
VOLUME_DEPTH_PADDED = 160    # original 155, padded to nearest multiple of 8 for U-Net skip connections

# Training
DEFAULT_BATCH_SIZE = 1
DEFAULT_LEARNING_RATE = 1e-4