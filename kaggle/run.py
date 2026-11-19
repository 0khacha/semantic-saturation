"""Kaggle entry point: clone SSB, install, then build / run / report one config on a T4.

Push with `kaggle kernels push -p kaggle --accelerator NvidiaTeslaT4`. Everything lands in
/kaggle/working/ssb and is saved as the notebook's output.
"""

import os
import subprocess
import sys

CONFIG = "configs/main.yaml"
STEPS = ["build", "run", "report"]  # e.g. ["run"] to resume after a timeout
REPO = "https://github.com/0khacha/semantic-saturation.git"
WORK = "/kaggle/working/ssb"


def sh(cmd: str) -> None:
    print(f"\n$ {cmd}", flush=True)
    code = subprocess.run(cmd, shell=True).returncode
    if code:
        print(f"command failed with exit code {code}", flush=True)
        sys.exit(code)


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402

print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available(), flush=True)
if not torch.cuda.is_available():
    sys.exit("No GPU in this session: choose Accelerator = GPU T4 (phone-verified account).")
print("GPU:", torch.cuda.get_device_name(0), flush=True)
if os.path.isdir(WORK):
    sh(f"git -C {WORK} pull --ff-only")
else:
    sh(f"git clone --depth 1 {REPO} {WORK}")
os.chdir(WORK)
sh("git log -1 --format='commit %h %s'")
sh("pip install -q -e '.[analysis]'")
for step in STEPS:
    extra = " --readme" if step == "report" else ""
    sh(f"ssb {step} --config {CONFIG}{extra}")
sh("ls -la data results/* figures | head -60")
