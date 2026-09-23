import torch

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用性: {torch.cuda.is_available()}")
print(f"当前设备: {torch.cuda.current_device()}")
print(f"设备名称: {torch.cuda.get_device_name(0)}")
print(f"设备数量: {torch.cuda.device_count()}")
print(f"CUDA版本: {torch.version.cuda}")
print(f"cuDNN版本: {torch.backends.cudnn.version()}")
