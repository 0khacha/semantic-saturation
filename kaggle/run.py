"""Kaggle entry point: clone SSB, install, run one config end to end.

Push with `kaggle kernels push -p kaggle --accelerator NvidiaTeslaT4`. The config comes
from SSB_CONFIG below; everything lands in /kaggle/working/ssb (saved as kernel output).
"""

import os
import subprocess
import sys

SSB_CONFIG = "configs/smoke.yaml"
EXTRA_RUN_ARGS = ""  # e.g. "--time-limit-hours 11" for the main run
REPO = "https://github.com/0khacha/semantic-saturation.git"
WORK = "/kaggle/working/ssb"


def sh(cmd: str) -> None:
    print(f"\n$ {cmd}", flush=True)
    code = subprocess.run(cmd, shell=True).returncode
    if code:
        print(f"command failed with exit code {code}", flush=True)
        sys.exit(code)


import torch  # noqa: E402

print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available(), flush=True)
if not torch.cuda.is_available():
    print("No GPU in this session. On Kaggle: verify your phone number (Settings), check the "
          "weekly GPU quota, and choose Accelerator = GPU T4.", flush=True)
    sys.exit(1)
print("GPU:", torch.cuda.get_device_name(0), flush=True)
if os.path.isdir(WORK):
    sh(f"git -C {WORK} pull --ff-only")
else:
    sh(f"git clone --depth 1 {REPO} {WORK}")
os.chdir(WORK)
sh("git log -1 --format='commit %h %s'")
sh("python --version")
sh("pip install -q -e '.[run,analysis]'")
sh(f"ssb run --config {SSB_CONFIG} {EXTRA_RUN_ARGS}")
sh(f"ssb report --config {SSB_CONFIG}")
sh("ls -R results figures | head -50")
