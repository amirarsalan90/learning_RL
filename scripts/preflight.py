#!/usr/bin/env python3
"""Read-only diagnostics; --require-cuda exits nonzero if GPU training is unavailable."""

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    report = {"python": platform.python_version(), "executable": sys.executable, "packages": {}}
    for package in ("torch", "transformers", "trl", "peft", "datasets", "accelerate", "jupyterlab"):
        try:
            report["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][package] = None
    cuda_available = False
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        report["cuda"] = {"available": cuda_available, "torch_cuda_version": torch.version.cuda, "devices": []}
        for index in range(torch.cuda.device_count() if cuda_available else 0):
            device = torch.cuda.get_device_properties(index)
            free, total = torch.cuda.mem_get_info(index)
            report["cuda"]["devices"].append({"name": device.name, "free_bytes": free, "total_bytes": total})
    except Exception as error:
        report["cuda"] = {"available": False, "error": str(error)}
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv"],
                                    capture_output=True, text=True, timeout=10)
            report["nvidia_smi"] = {"exit_code": result.returncode, "output": (result.stdout + result.stderr).strip()}
        except subprocess.TimeoutExpired:
            report["nvidia_smi"] = {"error": "timed out after 10 seconds"}
    print(json.dumps(report, indent=2))
    return 1 if args.require_cuda and not cuda_available else 0


if __name__ == "__main__":
    sys.exit(main())
