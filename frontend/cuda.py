import torch

def check_cuda():
    if torch.cuda.is_available():
        print("GPU is available")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("GPU is not available")

if __name__ == "__main__":
    check_cuda()
