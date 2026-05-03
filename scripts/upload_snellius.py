# File upload utility -- can run in parallel to preprocessing script, uploading new files as they are created.

import time
from pathlib import Path
from subprocess import run
import argparse
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--local", type=Path, required=True)
parser.add_argument("--remote", type=Path, required=True)
parser.add_argument("--host", type=str, default="snellius-dl2")
args = parser.parse_args()

# Check paths exist
assert args.local.exists(), f"No such path {args.local}"
run(["ssh", args.host, "ls", args.remote]).check_returncode()

processed = set()

print("Starting file search")
while True:
    new_files = [f for f in args.local.glob("*.h5") if f not in processed]
    if new_files:
        print(f"found {len(new_files)} files...")
    else:
        print(f"no files found ({datetime.datetime.now()})")
    # Wait for files to be saved properly
    time.sleep(60)
    for file in new_files:
        run(["scp", file, f"{args.host}:{args.remote / file.name}"]).check_returncode()
    processed.update(new_files)
    if new_files:
        print("\tuploaded.")
