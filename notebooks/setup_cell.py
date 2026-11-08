# ---- effsql setup cell: identical in every notebook (source: notebooks/setup_cell.py) ----
REPO_URL = "https://github.com/selimsidan/correct-is-not-enough"
REPO_DIR = "/content/correct-is-not-enough"
REF = "main"  # branch, tag or commit hash to run
GPU = False  # set True in GPU notebooks: installs the [gen] extra (vLLM)
DRIVE_ROOT = "/content/drive/MyDrive/586"
LOCAL_ROOT = "/content/bird_local"

import os, subprocess, sys
from google.colab import drive

drive.mount("/content/drive")


def sh(*cmd, check=True):
    r = subprocess.run(cmd, text=True, capture_output=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{r.stderr}")
    return r.stdout.strip()


if not os.path.isdir(REPO_DIR):
    sh("git", "clone", "--quiet", REPO_URL, REPO_DIR)
sh("git", "-C", REPO_DIR, "fetch", "--quiet", "--all", "--tags")
sh("git", "-C", REPO_DIR, "checkout", "--quiet", REF)
if sh("git", "-C", REPO_DIR, "rev-parse", "--verify", "--quiet", f"origin/{REF}", check=False):
    sh("git", "-C", REPO_DIR, "merge", "--ff-only", "--quiet", f"origin/{REF}")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-e", REPO_DIR + ("[gen]" if GPU else "")], check=True
)
os.environ["EFFSQL_DRIVE_ROOT"] = DRIVE_ROOT
os.environ["EFFSQL_LOCAL_ROOT"] = LOCAL_ROOT
print("effsql commit:", sh("git", "-C", REPO_DIR, "rev-parse", "HEAD"))
