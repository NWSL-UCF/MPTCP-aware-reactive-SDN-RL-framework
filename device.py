import torch

# Check CUDA availability
print("CUDA available:", torch.cuda.is_available())

# If available, check CUDA version and device details
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("Current CUDA device:", torch.cuda.current_device())
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("CUDA is not available on this system.")
