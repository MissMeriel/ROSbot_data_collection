import shutil
import sys
import os
from pathlib import Path

import shutil, errno

def copyanything(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno in (errno.ENOTDIR, errno.EINVAL):
            shutil.copy(src, dst)
        else: raise


src = "./training-output/"
dest = "/p/rosbot/rosbotxl/models4yili/"

dirs = os.listdir(src)
copied_dirs = os.listdir(dest)
dirs = [d for d in dirs if os.path.isdir(src + d)]
for d in dirs:
    if d not in copied_dirs:
        files = os.listdir(src + d)
        slurm_out_present = sum(["metainfo" in f for f in files])
        if slurm_out_present > 0:
            print(f"Copying {d} to {dest}")
            copyanything(src+d, dest)