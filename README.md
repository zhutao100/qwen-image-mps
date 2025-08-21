## Qwen Image (MPS/CUDA/CPU)

Generate and edit images from text prompts using the Hugging Face Diffusers pipeline for `Qwen/Qwen-Image`, with automatic device selection for Apple Silicon (MPS), NVIDIA CUDA, or CPU fallback.

### Features
- **Auto device selection**: prefers MPS (Apple Silicon), then CUDA, else CPU
- **Simple CLI**: provide a prompt and number of steps
- **Image generation**: create new images from text prompts
- **Image editing**: modify existing images using text instructions
- **Fast editing**: 8-step and 4-step editing using Lightning LoRA
- **Timestamped outputs**: avoids overwriting previous generations
- **Fast mode**: 8-step generation using Lightning LoRA (auto-downloads if needed)
- **Ultra-fast mode**: 4-step generation using Lightning LoRA (auto-downloads if needed)
 - **Multi-image generation**: generate multiple images in one run with `--num-images`

### Example

Example result you can create with this project:

![Example image](https://raw.githubusercontent.com/ivanfioravanti/qwen-image-mps/main/example.png)

## Installation

### Option 1: Install from PyPI (Recommended)

Install the package using pip:
```bash
pip install qwen-image-mps
```

Then run it directly from the command line:
```bash
qwen-image-mps --help
qwen-image-mps --version        # Show version
qwen-image-mps generate --help  # For image generation
qwen-image-mps edit --help      # For image editing
```

### Option 2: Direct script execution with uv

You can run this script directly using `uv run` without installation - it will install all dependencies automatically in an isolated environment:
```bash
uv run https://raw.githubusercontent.com/ivanfioravanti/qwen-image-mps/refs/heads/main/qwen-image-mps.py --help
```

Or download the file first:
```bash
curl -O https://raw.githubusercontent.com/ivanfioravanti/qwen-image-mps/refs/heads/main/qwen-image-mps.py
uv run qwen-image-mps.py --help
```

### Option 3: Install from source

Clone the repository and install in development mode:
```bash
git clone https://github.com/ivanfioravanti/qwen-image-mps.git
cd qwen-image-mps
pip install -e .
```

**Note:** The first time you run the tool, it will download the 57.7GB model from Hugging Face and store it in your `~/.cache/huggingface/hub/models--Qwen--Qwen-Image` directory.

## Usage

After installation, use the `qwen-image-mps` command with either `generate` or `edit` subcommands:

```bash
qwen-image-mps --help
qwen-image-mps --version        # Show version
qwen-image-mps generate --help  # For image generation
qwen-image-mps edit --help      # For image editing
```

### Image Generation Examples:

```bash
# Default prompt and steps
qwen-image-mps generate

# Custom prompt and fewer steps
qwen-image-mps generate -p "A serene alpine lake at sunrise, ultra detailed, cinematic" -s 30

# Fast mode with Lightning LoRA (8 steps)
qwen-image-mps generate -f -p "A magical forest with glowing mushrooms"

# Ultra-fast mode with Lightning LoRA (4 steps)
qwen-image-mps generate --ultra-fast -p "A magical forest with glowing mushrooms"
# Or use the short form
qwen-image-mps generate -uf -p "A magical forest with glowing mushrooms"


# Custom seed for reproducible generation
qwen-image-mps generate --seed 42 -p "A vintage coffee shop"

# Generate multiple images (incrementing seed per image when seed is provided)
qwen-image-mps generate -p "Retro sci-fi city skyline at night" --num-images 3 --seed 100

# Generate multiple images with a fresh random seed for each image (omit --seed)
qwen-image-mps generate -p "Retro sci-fi city skyline at night" --num-images 3
```

### Image Editing Examples:

```bash
# Basic image editing
qwen-image-mps edit -i input.jpg -p "Change the sky to sunset colors"

# Fast mode with Lightning LoRA (8 steps)
qwen-image-mps edit -i photo.png -p "Add snow to the mountains" --fast

# Ultra-fast mode with Lightning LoRA (4 steps)
qwen-image-mps edit -i landscape.jpg -p "Make it autumn colors" --ultra-fast
# Or use the short form
qwen-image-mps edit -i landscape.jpg -p "Make it autumn colors" -uf

# Edit with custom output filename
qwen-image-mps edit -i portrait.jpg -p "Change hair color to blonde" -o blonde_portrait.png

# Edit with custom seed and steps
qwen-image-mps edit -i scene.jpg -p "Add dramatic lighting" --seed 123 -s 30
```

If using the direct script with uv, replace `qwen-image-mps` with `uv run qwen-image-mps.py` in the examples above.

### Command Arguments

#### Generate Command Arguments
- `-p, --prompt` (str): Prompt text for image generation.
- `-s, --steps` (int): Number of inference steps (default: 50).
- `-f, --fast`: Enable fast mode using Lightning LoRA for 8-step generation.
- `-uf, --ultra-fast`: Enable ultra-fast mode using Lightning LoRA v1.0 for 4-step generation.
- `--seed` (int): Random seed for reproducible generation (default: 42). If not
  explicitly provided and generating multiple images, a new random seed is used
  for each image.
- `--num-images` (int): Number of images to generate (default: 1).

#### Edit Command Arguments
- `-i, --input` (str): Path to the input image to edit (required).
- `-p, --prompt` (str): Editing instructions (required).
- `-s, --steps` (int): Number of inference steps for normal editing (default: 50).
- `-f, --fast`: Enable fast mode using Lightning LoRA v1.1 for 8-step editing.
- `-uf, --ultra-fast`: Enable ultra-fast mode using Lightning LoRA v1.0 for 4-step editing.
- `--seed` (int): Random seed for reproducible generation (default: 42).
- `-o, --output` (str): Output filename (default: edited-<timestamp>.png).

## What the script does

### Image Generation
- Loads `Qwen/Qwen-Image` via `diffusers.DiffusionPipeline`
- Selects device and dtype:
  - MPS: `bfloat16`
  - CUDA: `bfloat16`
  - CPU: `float32`
- Uses a light positive conditioning suffix for quality
- Generates at a 16:9 resolution (default `1664x928`)
- Saves the output as `image-YYYYMMDD-HHMMSS.png` for a single image,
  or `image-YYYYMMDD-HHMMSS-1.png`, `image-YYYYMMDD-HHMMSS-2.png`, ... when using `--num-images`
- Prints the full path of the saved image

### Image Editing
- Loads `Qwen/Qwen-Image-Edit` via `QwenImageEditPipeline` for image editing
- Takes an existing image and editing instructions as input
- Applies transformations while preserving the original structure
- Saves the edited image as `edited-YYYYMMDD-HHMMSS.png` or custom filename
- Prints the full path of the edited image

### Fast Mode & Ultra-Fast Mode (Lightning LoRA)

#### Fast Mode (`-f/--fast`)
When using the `-f/--fast` flag, the tool:
- Automatically downloads the Lightning LoRA v1.1 from Hugging Face (cached in `~/.cache/huggingface/hub/`)
- Merges the LoRA weights into the model for accelerated generation
- Uses fixed 8 inference steps with CFG scale 1.0
- Provides ~6x speedup compared to the default 50 steps

#### Ultra-Fast Mode (`-uf/--ultra-fast`)
When using the `-uf/--ultra-fast` flag, the tool:
- Automatically downloads the Lightning LoRA v1.0 from Hugging Face (cached in `~/.cache/huggingface/hub/`)
- Merges the LoRA weights into the model for maximum speed generation
- Uses fixed 4 inference steps with CFG scale 1.0
- Provides ~12x speedup compared to the default 50 steps
- Ideal for rapid prototyping and iteration

The fast implementation is based on [Qwen-Image-Lightning](https://github.com/ModelTC/Qwen-Image-Lightning). The Lightning LoRA models are available on HuggingFace at [lightx2v/Qwen-Image-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Lightning).

Both generation and editing now support Lightning LoRA for accelerated processing!

## Notes and tweaks
- **Aspect ratio / resolution**: The script currently uses the `16:9` entry from an `aspect_ratios` map. You can change the selection in the code where `width, height` is set.
- **Determinism**: Use the `--seed` parameter to control the random generator for reproducible results. On MPS, the random generator runs on CPU for improved stability.
- **Performance**: If you hit memory or speed issues, try reducing `--steps`.

## Troubleshooting
- If you see "Using CPU" in the console on Apple Silicon, ensure your PyTorch build includes MPS and you are running on Apple Silicon Python (not under Rosetta).
- If model download fails or is unauthorized, log in with `huggingface-cli login` or accept the model terms on the Hugging Face model page.

## Development

To contribute or modify the tool:

1. Clone the repository:
```bash
git clone https://github.com/ivanfioravanti/qwen-image-mps.git
cd qwen-image-mps
```

2. Install in development mode with dev dependencies:
```bash
pip install -e ".[dev]"
```

3. Install pre-commit hooks:
```bash
pre-commit install
```

The project uses:
- `black` for code formatting
- `isort` for import sorting
- `ruff` for linting
- Pre-commit hooks for code quality

## Repository contents
- `src/qwen_image_mps/`: Main package source code
- `qwen-image-mps.py`: Script wrapper for direct URL execution
- `pyproject.toml`: Package configuration and dependencies
- `uv.lock`: Locked dependencies for reproducible builds
- `.github/workflows/`: CI/CD pipelines for testing and publishing
- `example.png`: Sample generated image
