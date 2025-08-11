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

import importlib.util
import os
import sys
import tempfile
import urllib.request


def import_from_url(url, module_name):
    """Import a Python module from a URL."""
    with urllib.request.urlopen(url) as response:
        code = response.read()

    # Create a temporary file to hold the module
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".py", delete=False) as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    try:
        # Load the module from the temporary file
        spec = importlib.util.spec_from_file_location(module_name, tmp_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        # Clean up the temporary file
        os.unlink(tmp_path)


if __name__ == "__main__":
    # Import cli.py from GitHub
    cli_url = "https://raw.githubusercontent.com/ivanfioravanti/qwen-image-mps/refs/heads/test-script-fix/src/qwen_image_mps/cli.py"
    cli = import_from_url(cli_url, "qwen_image_mps_cli")

    # Run the main function
    cli.main()
