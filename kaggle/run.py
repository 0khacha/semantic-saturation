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


sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv")
if os.path.isdir(WORK):
    sh(f"git -C {WORK} pull --ff-only")
else:
    sh(f"git clone --depth 1 {REPO} {WORK}")
os.chdir(WORK)
sh("git log -1 --format='commit %h %s'")
sh("python --version && python -c 'import torch; print(\"torch\", torch.__version__)'")
sh("pip install -q -e '.[run,analysis]'")
sh(f"ssb run --config {SSB_CONFIG} {EXTRA_RUN_ARGS}")
sh(f"ssb report --config {SSB_CONFIG}")
sh("ls -R results figures | head -50")
