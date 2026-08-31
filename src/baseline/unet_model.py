import torch
from monai.networks.nets import UNet

def build_model():
    """
    Real 3D U-Net for BraTS segmentation.
    in_channels=4: the four MRI modalities (t1, t1ce, t2, flair) stacked as channels.
    out_channels=4: background + 3 tumor sub-region classes (label 4 remapped to index 3 later).
    channels/strides: 4 down/up steps, doubling feature channels each level (16->32->64->128).
    num_res_units=2: each block uses residual connections (helps gradients flow in deep nets).
    """
    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
    )
    return model

if __name__ == "__main__":
    model = build_model()

    # Count learnable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Sanity check: run a fake batch through it, same pattern as TinyUNet3D
    fake_input = torch.rand(1, 4, 64, 64, 64)  # small spatial size to keep this fast
    output = model(fake_input)
    print(f"\nInput shape: {fake_input.shape}")
    print(f"Output shape: {output.shape}")