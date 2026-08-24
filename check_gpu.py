import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
    # The real test: try an actual GPU computation, not just detection
    x = torch.rand(1000, 1000, device="cuda")
    y = torch.rand(1000, 1000, device="cuda")
    z = x @ y
    torch.cuda.synchronize()
    print("Matrix multiply on GPU: SUCCESS")
else:
    print("No CUDA GPU detected by PyTorch")