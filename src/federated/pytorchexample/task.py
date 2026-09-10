"""pytorchexample: FedMed - real 3D U-Net + BraTS data for federated training."""

import os
import sys

import torch
from torch.utils.data import DataLoader

# NOTE: hardcoded because Flower's ServerApp runs from an isolated copy of
# this app (installed to ~/.flwr/apps/...), so a __file__-relative path
# computes the wrong location. Fragile if the project ever moves machines -
# acceptable tradeoff given the deadline, revisit if this becomes a real
# multi-machine deployment.
sys.path.insert(0, r"C:\FedMed\src\baseline")

from unet_model import build_model
from brats_dataset import BraTSDataset
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.utils import one_hot


def Net():
    """Wraps build_model() so the rest of this file can call Net() like the
    original quickstart did, keeping the ClientApp/ServerApp code unchanged."""
    return build_model()


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Load this node's slice of BraTS patients - a real data partition,
    not CIFAR. Patients 1-15 split evenly across num_partitions nodes."""
    all_patient_ids = [f"BraTS20_Training_{i:03d}" for i in range(1, 16)]

    # Simple even split: partition_id 0 gets patients [0:5], 1 gets [5:10], etc.
    per_partition = len(all_patient_ids) // num_partitions
    start = partition_id * per_partition
    end = start + per_partition
    my_patients = all_patient_ids[start:end]

    # 80/20 train/test split within this node's own patients
    split = max(1, int(len(my_patients) * 0.8))
    train_ids, test_ids = my_patients[:split], my_patients[split:] or my_patients[-1:]

    trainloader = DataLoader(BraTSDataset(train_ids), batch_size=batch_size, shuffle=True)
    testloader = DataLoader(BraTSDataset(test_ids), batch_size=batch_size)
    return trainloader, testloader


def train(net, trainloader, epochs, lr, device):
    """Train using DiceCELoss, same as our baseline train.py."""
    net.to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    running_loss = 0.0
    steps = 0
    for _ in range(epochs):
        for image, mask in trainloader:
            image, mask = image.to(device), mask.to(device).unsqueeze(1)
            optimizer.zero_grad()
            output = net(image)
            loss = loss_fn(output, mask)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            steps += 1
    return running_loss / max(steps, 1)


def test(net, testloader, device):
    """Evaluate using Dice score, same metric as our baseline."""
    net.to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    net.eval()
    total_loss, steps = 0.0, 0
    with torch.no_grad():
        for image, mask in testloader:
            image, mask = image.to(device), mask.to(device).unsqueeze(1)
            output = net(image)
            total_loss += loss_fn(output, mask).item()
            steps += 1
            pred = torch.argmax(output, dim=1, keepdim=True)
            dice_metric(y_pred=one_hot(pred, num_classes=4), y=one_hot(mask, num_classes=4))
    dice_score = dice_metric.aggregate().item()
    dice_metric.reset()
    return total_loss / max(steps, 1), dice_score