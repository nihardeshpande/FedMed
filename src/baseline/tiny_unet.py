import torch
import torch.nn as nn

class TinyUNet3D(nn.Module):
    """
    A minimal 3D U-Net: one downsampling step, one upsampling step.
    Real U-Nets have 4-5 of these steps; this is a toy version to prove
    the encoder-decoder pattern works before scaling up.
    """
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        # ENCODER: a conv layer, then downsample (shrink spatial size, learn features)
        self.enc_conv = nn.Conv3d(in_channels, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2)  # halves depth, height, width

        # DECODER: upsample back to original size, then a conv to produce the mask
        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.dec_conv = nn.Conv3d(8, out_channels, kernel_size=3, padding=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: [batch, 1, D, H, W]
        x = self.enc_conv(x)     # [batch, 8, D, H, W]  -- more feature channels, same spatial size
        x = self.relu(x)
        x = self.pool(x)         # [batch, 8, D/2, H/2, W/2] -- shrunk

        x = self.upsample(x)     # [batch, 8, D, H, W] -- back to original spatial size
        x = self.dec_conv(x)     # [batch, 1, D, H, W] -- back to 1 channel = the predicted mask

        return x

if __name__ == "__main__":
    model = TinyUNet3D()
    print(model)

    fake_input = torch.rand(2, 1, 32, 64, 64)
    output = model(fake_input)
    print("\nInput shape:", fake_input.shape)
    print("Output shape:", output.shape)