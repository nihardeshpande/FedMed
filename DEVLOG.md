# FedMed Devlog

## Phase 0 — Environment Setup

**Date:** 2026-08-24

- Attempted WSL2 setup for Linux-based dev environment (motivated by TenSEAL's
  poor Windows compilation support). Hit a corrupted Windows Installer /
  COM registration issue (`REGDB_E_CLASSNOTREG`) that survived a feature
  re-enable + reboot. Decided to abandon WSL2 for now and build natively
  on Windows instead, accepting the risk that TenSEAL (Week 3 of the spec)
  may need a workaround later (separate Python 3.12 env, Docker, or revisiting
  WSL once the Installer issue is fixed).
- System Python is 3.14.6. Verified PyTorch 2.11 officially supports 3.14
  on Windows, but TenSEAL only ships wheels up to Python 3.13 — noted as a
  known future risk, deferred until Week 3.
- Created project venv (`python -m venv venv`) to isolate dependencies.
- GPU: NVIDIA RTX 5050 Laptop (Blackwell, compute capability sm_120).
  Verified real GPU compute (not just detection) works via a CUDA matmul
  smoke test using `torch==2.11.0+cu128`.


  ## Week A, Day 1 — Mon Aug 31 (worked Sat Aug 29)

- Team plan finalized. Role clarified as 1a (ML Core), paired with Abhi (1b, Flower
  orchestration). Kept `TinyUNet3D` (hand-rolled) as a learning artifact from Phase 0
  practice — the real Week 1 deliverable will use `monai.networks.nets.UNet` instead,
  per the spec.
- Obtained BraTS2020 via Kaggle mirror (`awsaf49/brats20-dataset-training-validation`)
  instead of the official CBICA portal, to avoid registration delays. Note: cite the
  official BraTS papers in any report, not the Kaggle page. Folder structure has an
  extra nested wrapper folder that isn't obvious from the dataset name:
  `data/brats2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/<patient_id>/`
  — flagging for Charan (Role 4) so he doesn't rediscover this separately.
- Wrote `load_patient.py`, loads all 5 files (t1, t1ce, t2, flair, seg) for one patient
  using nibabel. Verified: shape (240,240,155) consistent across all modalities/mask,
  BraTS's known segmentation label set [0,1,2,4] confirmed present, dtype float64
  (will need cast to float32 before training). Raw scanner intensities are unnormalized
  and vary wildly per modality (T1CE up to 1845) — TODO before training: normalize
  each modality.
- Full BraTS2020 download is ~26GB uncompressed (240x240x155 voxels x 5 files x 369
  patients, stored as .nii not .nii.gz). Decided not to worry about trimming this down
  now since GPU VRAM (8GB) forces small-batch training regardless; will use 5-10
  patient subsets for pipeline development (per Wed's task) and a larger subset for
  the real baseline pass (Thu), not necessarily all 369.
- Hit a GitHub ruleset gap: branch protection required 1 approval, but did NOT exempt
  repo admins from approving their own PRs by default — this blocked merging solo
  work entirely. Fixed by adding "Repository admin" to the ruleset's bypass list
  (Settings > Rules > Rulesets > Bypass list), scoped to "Always". Teammates still
  require review; only admin-authored PRs skip it.
- Merged PR #1 (`ml/data-loading`) into main.