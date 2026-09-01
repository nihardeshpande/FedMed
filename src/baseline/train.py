import torch
from torch.utils.data import DataLoader
from monai.losses import DiceCELoss

from unet_model import build_model
from brats_dataset import BraTSDataset

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Small subset for today - just proving the loop runs correctly
    patient_ids = [f"BraTS20_Training_{i:03d}" for i in range(1, 6)]  # 5 patients
    dataset = BraTSDataset(patient_ids)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = build_model().to(device)

    # softmax=True: applies softmax to the model's raw logits internally before comparing to target.
    # to_onehot_y=True: converts the integer mask [240,240,155] into one-hot [4,240,240,155]
    # internally, so it can be compared against the model's 4-channel output per class.
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()  # puts the model in "training mode" (matters for certain layers, not all)

    for epoch in range(1):  # just 1 epoch today, to confirm it runs
        epoch_loss = 0.0
        for step, (image, mask) in enumerate(loader):
            image = image.to(device)
            mask = mask.to(device).unsqueeze(1)  # add channel dim: [1,240,240,155] -> [1,1,240,240,155]

            optimizer.zero_grad()          # clear gradients from the previous step
            output = model(image)           # forward pass
            loss = loss_fn(output, mask)    # compute how wrong we are
            loss.backward()                 # compute gradients via autograd
            optimizer.step()                # nudge weights based on gradients

            epoch_loss += loss.item()
            print(f"  Step {step+1}/{len(loader)} - loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1} average loss: {epoch_loss/len(loader):.4f}")

if __name__ == "__main__":
    main()