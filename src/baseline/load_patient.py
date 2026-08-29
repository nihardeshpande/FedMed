import nibabel as nib
import numpy as np

PATIENT_DIR = r"data\brats2020\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001"

def load_modality(parient_dir, modality):
    """
    Loads one MRI modality file (e.g. 't1', 'flair', 'seg') for a patient.
    Returns a NumPy as array of the voxel data.
    """

    filepath = f"{PATIENT_DIR}\\BraTS20_Training_001_{modality}.nii"
    img = nib.load(filepath)   # reads the file + metadata, doesn't load pixel data into memory yet
    data = img.get_fdata()  # NOW it actually loads the voxel array, as float64 by default
    return data

if __name__ == "__main__":
    modalities = ["t1","t1ce","t2","flair","seg"]

    for mod in modalities:
        data = load_modality(PATIENT_DIR, mod)
        print(f"\n{mod.upper()}")
        print(f"    Shape: {data.shape}")
        print(f"    Dtype: {data.dtype}")
        print(f"    Value range: [{data.min(): .2f}, {data.max():.2f}]")
        if mod == "seg":
            print(f"    Unique labels: {np.unique(data)}")