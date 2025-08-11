# /// script
# dependencies = [
#   "torch",
#   "diffusers @ git+https://github.com/huggingface/diffusers",
#   "transformers",
#   "accelerate",
#   "safetensors",
# ]
# ///

"""
This script wrapper allows direct execution via:
uv run https://raw.githubusercontent.com/ivanfioravanti/qwen-image-mps/refs/heads/main/qwen-image-mps.py

For package installation, use: pip install qwen-image-mps
"""

from src.qwen_image_mps.cli import main

if __name__ == "__main__":
    main()
