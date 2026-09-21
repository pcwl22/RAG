"""Print the active Python/PyTorch CUDA runtime and run a tiny GPU operation."""
from __future__ import annotations

import json
import sys
from typing import Any

import torch


def main() -> None:
    available = torch.cuda.is_available()
    result: dict[str, Any] = {
        "python": sys.version,
        "executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": available,
        "device_count": torch.cuda.device_count(),
        "device": torch.cuda.get_device_name(0) if available else None,
    }
    if available:
        tensor = torch.tensor([1.0, 2.0], device="cuda")
        result["gpu_smoke_sum"] = float(tensor.sum().cpu())
        result["allocated_bytes"] = torch.cuda.memory_allocated()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
