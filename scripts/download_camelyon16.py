#!/usr/bin/env python3
"""
Download script for the CAMELYON16 dataset.

This script uses `wget` to download the public, authentication-free mirroring
of CAMELYON16 hosted by GigaScience (GigaDB FTP).

This is vastly more reliable than the Grand Challenge API which frequently
requires refreshed API tokens and complex CLI abstraction.

Usage:
    python scripts/download_camelyon16.py --output_dir datasets/camelyon16
"""

import argparse
import os
import subprocess
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def download_camelyon16(output_dir: Path):
    """Download CAMELYON16 using wget from GigaDB."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Downloading CAMELYON16 to {output_dir.absolute()} ...")
    logging.info("This will take a VERY long time (approx 400 GBs). Progress will be printed below.")
    
    # Official GigaScience mirror for CAMELYON16 dataset 100439
    ftp_url = "ftp://parrot.genomics.cn/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/"
    
    try:
        # wget -r (recursive) -nH (no host directories) --cut-dirs=5 (remove up to 100439/) 
        # --no-parent (don't ascend FTP) 
        # -c (continue/resume partially downloaded files)
        cmd = [
            "wget", "-r", "-nH", "--cut-dirs=5", "--no-parent", "-c",
            "-P", str(output_dir),
            ftp_url
        ]
        
        # We execute wget allowing output to stream directly to the terminal so the user sees progress bars
        subprocess.run(cmd, check=True)
        logging.info("Download completed successfully!")
        
    except FileNotFoundError:
        logging.error("The 'wget' command was not found on your system. Please install it (e.g. `sudo apt install wget`).")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logging.error(f"Download failed with error: {e}")
        logging.info("You can resume the download anytime by running this script again. 'wget' will automatically resume partial files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download CAMELYON16 dataset.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/camelyon16",
        help="Directory to save the downloaded images."
    )
    args = parser.parse_args()
    
    download_camelyon16(Path(args.output_dir))

