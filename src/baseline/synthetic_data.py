import torch

def get_fake_mri_batch(batch_size =2, depth = 32, height = 64, width = 64):
    """Generates a fake batch of 3D MRI-shaped data for testing the training
    pipeline before we touch real BraTS data.
    
    Real MRI volumes are 3D grids of intensity values(Like a stack of 2D slices)
    . We're faking that shape here: [batch, channels, height, width].
    channels = 1 because we're pretending it's a single MRI modality(e.g just T1).
    """

    # Fake input volume: random floats simulating scan intensities
    images = torch.rand(batch_size, 1, depth, height, width)

        # Fake ground-truth tumor mask: random binary values (0 = healthy, 1 = tumor)
    # In real BraTS data, this comes from expert-labeled segmentation maps.

    masks = torch.randint(0,2, (batch_size, 1, depth, height, width)).float()

    return images, masks

if __name__ == "__main__":
    images, masks = get_fake_mri_batch()
    print("Images shape:", images.shape)
    print("Images dtype:", images.dtype)
    print("Masks shape:", masks.shape)
    print("Masks dtype:", masks.dtype)
    print("Unique mask values:", torch.unique(masks))