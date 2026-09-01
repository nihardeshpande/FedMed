# FedMed Devlog

## Aug 20 — Project Specifications pdf
- Added the project spec document to the repo for reference.

## Aug 24 — Phase 0
- Attempted WSL2 setup for a Linux-based dev environment (motivated by TenSEAL's
  poor Windows compilation support). Hit a corrupted Windows Installer / COM
  registration issue (`REGDB_E_CLASSNOTREG`) that survived a feature re-enable +
  reboot. Abandoned WSL2, built natively on Windows instead — accepted the risk
  that TenSEAL (encryption phase, later in the project) may need a workaround
  eventually (separate Python env, Docker, or revisiting WSL).
- System Python is 3.14.6. Verified PyTorch supports 3.14 on Windows, but flagged
  TenSEAL only ships wheels up to Python 3.13 as a known future risk.
- Created project venv to isolate dependencies.
- GPU: NVIDIA RTX 5050 Laptop (Blackwell, compute capability sm_120). Verified
  real GPU compute (not just detection) via a CUDA matmul smoke test using
  `torch==2.11.0+cu128`.

## Aug 25 — Add practice folder with tensor basics exploration / Update tensor basics practice
- Learned PyTorch tensor fundamentals: creation, shape/dtype, moving to GPU,
  element-wise vs. matrix multiply (`*` vs `@`).
- Key concept: models require `float32` tensors for autograd; integer tensors
  can't carry gradients.

## Aug 26 — Build 2-layer network with ReLU, debug shape mismatch and missing return / Add synthetic MRI data generator
- Built a small `nn.Module` (2 linear layers + ReLU) by hand. Hit and diagnosed
  two real bugs: a layer shape mismatch (`RuntimeError: mat1 and mat2 shapes
  cannot be multiplied`) caused by chaining layers with non-matching
  in/out feature counts, and a silent `None` output caused by a missing
  `return` statement in `forward()`.
- Wrote `synthetic_data.py`: generates fake MRI-shaped 5D tensors
  (`[batch, channels, D, H, W]`) to test the training pipeline without
  waiting on the real dataset download.

## Aug 27 — Add minimal 3D U-Net (single down/up step)
- Built `tiny_unet.py`: a toy 3D U-Net (one encoder step, one decoder step)
  by hand, using `Conv3d`/`MaxPool3d`/`Upsample`, to understand the
  encoder-decoder shape-matching pattern before using a library implementation.
  Verified output shape exactly matches input shape.

## Aug 29 — ml/data-loading: load single BraTS volume, add data/ to gitignore
- Team plan finalized. Role clarified as 1a (ML Core), paired with Abhi (1b,
  Flower orchestration).
- Obtained BraTS2020 via Kaggle mirror (`awsaf49/brats20-dataset-training-validation`)
  instead of the official CBICA portal, to avoid registration delays. Note: cite
  the official BraTS papers in any report, not the Kaggle page. Folder structure
  has an extra nested wrapper folder:
  `data/brats2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/<patient_id>/`
  — flagged for Charan (Role 4) to avoid re-discovering this separately.
- Wrote `load_patient.py`: loads all 5 files (t1, t1ce, t2, flair, seg) for one
  patient using nibabel. Verified shape `(240,240,155)` consistent across all
  modalities/mask, BraTS's known segmentation label set `[0,1,2,4]` confirmed
  present, dtype float64 (needs cast to float32 before training). Raw scanner
  intensities are unnormalized and vary wildly per modality (T1CE up to 1845) —
  flagged as a TODO before training.
- Full BraTS2020 download is ~26GB uncompressed. Decided not to worry about
  trimming this down since GPU VRAM (8GB) forces small-batch training regardless;
  planned to use 5-10 patient subsets for pipeline development, larger subset for
  the real baseline pass later.
- Hit a GitHub ruleset gap: branch protection required 1 approval but did NOT
  exempt repo admins from approving their own PRs by default — blocked merging
  solo work entirely. Fixed by adding "Repository admin" to the ruleset's bypass
  list (Settings > Rules > Rulesets > Bypass list, scoped to "Always").
  Teammates still require review; only admin-authored PRs skip it.

## Aug 31 — docs: log Day 1 progress / ml/model: define 3D U-Net architecture
- Logged Day 1 progress (folded into this file's structure above).
- Built the real 3D U-Net using `monai.networks.nets.UNet` (spatial_dims=3,
  in_channels=4 for the 4 MRI modalities, out_channels=4 for background + 3
  tumor sub-regions, channels=(16,32,64,128), strides=(2,2,2), num_res_units=2).
  ~1.19M parameters. Verified shape-preserving on a synthetic 64x64x64 input.
- Lost the first version of `unet_model.py` before committing it — ran and
  verified successfully, but moved to the next task without committing
  immediately. File was gone by the next session. Lesson: commit each file the
  moment it's verified working, don't batch across multiple tasks even when
  moving fast. Recreated and recommitted successfully.

## Sep 1 — ml/train-loop: DiceCELoss + Adam training loop, fix U-Net skip-connection padding bug
- Wrote `brats_dataset.py`: a proper `torch.utils.data.Dataset` subclass. Loads
  all 4 modalities per patient, z-score normalizes each independently, stacks
  as channels. Remaps segmentation label 4 -> 3 so labels are contiguous
  `[0,1,2,3]` for CrossEntropy/DiceCE loss compatibility.
- Wrote `train.py`: `DiceCELoss` (to_onehot_y=True, softmax=True) + Adam
  optimizer, real training loop (forward -> loss -> backward -> optimizer step).
- Hit a real architecture bug: U-Net skip connections require encoder/decoder
  tensor sizes to match exactly at each level. With `strides=(2,2,2)` (3
  downsampling steps, needs dims divisible by 2^3=8), BraTS's depth dimension
  (155) doesn't divide evenly (155/8=19.375) while height/width (240) do
  (240/8=30). Fixed by zero-padding depth 155 -> 160 (nearest multiple of 8)
  in `brats_dataset.py`.
- Result: training loop ran end-to-end on 5 real patients, 1 epoch. Loss
  decreased every step (2.4790 -> 2.3771) — first real confirmation the whole
  pipeline (load -> normalize -> pad -> batch -> forward -> loss -> backward ->
  step) works correctly, not just "runs without crashing."