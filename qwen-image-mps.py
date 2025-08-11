# /// script
# dependencies = [
#   "torch",
#   "diffusers @ git+https://github.com/huggingface/diffusers",
#   "transformers",
#   "accelerate",
#   "safetensors",
#   "huggingface-hub",
# ]
# ///

"""
This script wrapper allows direct execution via:
uv run https://raw.githubusercontent.com/ivanfioravanti/qwen-image-mps/refs/heads/main/qwen-image-mps.py

For package installation, use: pip install qwen-image-mps
"""

import os
import sys

# Try to import from installed package first
try:
    from qwen_image_mps.cli import main
except ImportError:
    # Fall back to local development import
    try:
        # Add the directory containing this script to the path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        src_path = os.path.join(script_dir, "src")
        if os.path.exists(src_path):
            sys.path.insert(0, src_path)
        from qwen_image_mps.cli import main
    except ImportError:
        # If we still can't import, try from src subdirectory
        from src.qwen_image_mps.cli import main

if __name__ == "__main__":
    main()
