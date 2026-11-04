"""Record the observed environment to results/m0/environment.json.

Usage: .venv/bin/python scripts/env_report.py
"""

import json
import platform
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import kvlab.model as km  # noqa: E402
import torch  # noqa: E402


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as e:  # recorded, not hidden
        return f"ERROR: {e}"


def main():
    os_release = dict(line.split("=", 1) for line in Path("/etc/os-release").read_text().splitlines() if "=" in line)
    disk = shutil.disk_usage(km.PROJECT_ROOT)
    meminfo = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    info = {
        "os": os_release.get("PRETTY_NAME", "").strip('"'),
        "kernel": platform.release(),
        "wsl": "microsoft" in platform.release().lower(),
        "python": sys.version.split()[0],
        "venv": sys.prefix,
        "packages": {p: version(p) for p in ["torch", "transformers", "tokenizers", "safetensors",
                                             "huggingface_hub", "numpy", "pytest"]},
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "nvidia_smi": run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version,compute_cap",
                           "--format=csv,noheader"]),
        "nvidia_smi_cuda_version": next((line.split("CUDA Version:")[1].split("|")[0].strip()
                                         for line in run(["nvidia-smi"]).splitlines() if "CUDA Version" in line),
                                        None),
        "disk_free_gib_project_fs": round(disk.free / 2**30, 1),
        "host_mem_total": meminfo["MemTotal"].strip(),
        "cpu_count": __import__("os").cpu_count(),
        "hf_home": __import__("os").environ["HF_HOME"],
        "model_id": km.MODEL_ID,
        "model_revision": km.MODEL_REVISION,
        "requested_dtype": str(km.DTYPE),
        "requested_attn_implementation": km.ATTN_IMPLEMENTATION,
    }
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_capability"] = list(torch.cuda.get_device_capability(0))
        info["gpu_mem_total_mib"], info["gpu_mem_free_mib"] = total >> 20, free >> 20
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
    out = km.PROJECT_ROOT / "results" / "m0" / "environment.json"
    out.write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
