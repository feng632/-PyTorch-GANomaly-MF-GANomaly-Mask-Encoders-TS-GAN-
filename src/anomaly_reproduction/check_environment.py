"""Small, safe environment check used before starting experiments."""

from __future__ import annotations

import platform

import torch


def main() -> None:
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA build: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise SystemExit("GPU check failed: PyTorch cannot access CUDA.")

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(device)}")

    # A real GPU calculation, not just a device-name lookup.
    left = torch.randn(512, 512, device=device)
    right = torch.randn(512, 512, device=device)
    result = left @ right
    torch.cuda.synchronize()

    print(f"Matrix result device: {result.device}")
    print(f"Matrix result shape: {tuple(result.shape)}")
    print("Environment check: PASS")


if __name__ == "__main__":
    main()

