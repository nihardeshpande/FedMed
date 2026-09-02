# FedMed

A privacy-preserving federated learning system for brain tumor segmentation.
Three simulated hospital nodes train a 3D U-Net locally on their own MRI data;
only encrypted, differentially-private model updates are shared with a central
server for aggregation — no raw patient data ever leaves a node.

## Stack

- **ML**: PyTorch, MONAI (3D U-Net, DiceCELoss, DiceMetric)
- **Federated orchestration**: Flower
- **Secure communication**: gRPC + TLS
- **Privacy**: TenSEAL (homomorphic encryption), differential privacy on weight updates
- **Dashboard**: React + Recharts, live metrics over WebSocket

## Dataset

[BraTS2020](https://www.med.upenn.edu/cbica/brats2020/data.html) — 369 patients,
4 MRI modalities per patient (T1, T1CE, T2, FLAIR) plus expert-labeled tumor
segmentation masks. Not included in this repo (~26GB) — see setup below.

## Project status

Currently mid-build. Centralized baseline (single 3D U-Net, no federation yet)
is working end-to-end: real BraTS data loads, normalizes, trains, and reports
a Dice score. Federated orchestration, encryption, and the dashboard are in
progress — see `DEVLOG.md` for the detailed build log and `FedMed_Dated_Plan.md`
for the team's task breakdown and timeline.

## Setup

### Environment

python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt


### Dataset
1. Create a free [Kaggle](https://kaggle.com) account, generate an API token
   (Settings → API → Create New Token), save as `~/.kaggle/kaggle.json`.
2. ```
   pip install kaggle
   mkdir data
   cd data
   kaggle datasets download -d awsaf49/brats20-dataset-training-validation
   Expand-Archive brats20-dataset-training-validation.zip -DestinationPath brats2020
Repo structure
src/
  baseline/          # centralized (non-federated) training pipeline
    load_patient.py    # loads one patient's NIfTI volumes via nibabel
    brats_dataset.py   # PyTorch Dataset: loads, normalizes, pads BraTS volumes
    unet_model.py       # 3D U-Net (monai.networks.nets.UNet)
    train.py            # training loop: DiceCELoss + Adam, logs loss + Dice score
    model_params.py     # get_parameters/set_parameters for Flower NumPyClient
    tiny_unet.py         # hand-rolled toy U-Net, kept as a learning artifact
practice/            # scratch/exploration files, not part of the pipeline
data/                # BraTS dataset (gitignored, not committed)
DEVLOG.md            # chronological build log: what, and why


Branching

Feature branches per role prefix (ml/, fed/, net/, crypto/, data/,
dash/), PR into main, regular merge commits (no squash) so individual
commit history stays visible for review.


**Create it:**
```powershell
notepad README.md
```
Paste the content, save.

Two things worth deciding before you commit:
1. **`requirements.txt` — is it current?** You generated one back in Phase 0, before MONAI, nibabel, and kaggle were installed. This README references it, so let's regenerate it now:
```powershell
   pip freeze > requirements.txt
```
2. Is `FedMed_Dated_Plan.md` actually a file in your repo, or just something you've been pasting into our chat? If it doesn't exist as a file yet, either add it (paste that plan into a new file) or I'll adjust the README's reference to it.

Answer #2, then paste `type README.md` and `type requirements.txt` (or the relevant diff) so I can confirm before we commit — same branch/PR pattern as everything else today.
