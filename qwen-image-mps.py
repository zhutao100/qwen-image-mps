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
Standalone script for Qwen-Image generation on Apple Silicon.
This script can be run directly via:
uv run https://raw.githubusercontent.com/ivanfioravanti/qwen-image-mps/refs/heads/main/qwen-image-mps.py

For package installation, use: pip install qwen-image-mps
"""

import argparse
import time
from datetime import datetime

import torch
from diffusers import FluxPipeline
from huggingface_hub import hf_hub_download


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an image with Qwen-Image",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default="""A coffee shop entrance features a chalkboard sign reading "Apple Silicon Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "Generated with MPS on Apple Silicon". Next to it hangs a poster showing a beautiful Italian woman, and beneath the poster is written "Just try it!". Ultra HD, 4K, cinematic composition""",
        help="Prompt text to condition the image generation.",
    )
    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=50,
        help="Number of inference steps for normal generation.",
    )
    parser.add_argument(
        "-f",
        "--fast",
        action="store_true",
        help="Use Lightning LoRA for fast generation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=195,
        help="Random seed for reproducible generation.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=1,
        help="Number of images to generate.",
    )
    return parser


def get_lora_path():
    """Get the Lightning LoRA from Hugging Face cache."""
    try:
        # This will download to HF cache or return cached path
        lora_path = hf_hub_download(
            repo_id="ByteDance/Hyper-SD",
            filename="Hyper-Flux.1-dev-8steps-lora.safetensors",
        )
        return lora_path
    except Exception as e:
        print(f"Warning: Could not download Lightning LoRA: {e}")
        return None


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        print("MPS is not available. This script requires Apple Silicon.")
        return

    print("Initializing Qwen-Image model...")
    pipe = FluxPipeline.from_pretrained(
        "Qwen/Qwen2-VL-Flux", torch_dtype=torch.bfloat16
    )

    # Apply Lightning LoRA if fast mode is requested
    if args.fast:
        lora_path = get_lora_path()
        if lora_path:
            print(f"Loading Lightning LoRA from: {lora_path}")
            pipe.load_lora_weights(lora_path)
            pipe.fuse_lora(lora_scale=1.0)
            steps = 8  # Lightning LoRA uses 8 steps
        else:
            print("Lightning LoRA not available, using normal generation.")
            steps = args.steps
    else:
        steps = args.steps

    pipe.to("mps")

    # Run once to warm up
    print("Warming up the model...")
    warm_up_prompt = "A test image"
    pipe(prompt=warm_up_prompt, num_inference_steps=1, width=256, height=256)

    # Generate images
    print(f"\nGenerating {args.num_images} image(s) with {steps} steps...")
    print(f"Prompt: {args.prompt}")

    generator = torch.Generator("mps").manual_seed(args.seed)

    for i in range(args.num_images):
        print(f"\nGenerating image {i+1}/{args.num_images}...")
        start_time = time.time()

        image = pipe(
            prompt=args.prompt,
            num_inference_steps=steps,
            width=1024,
            height=1024,
            generator=generator,
        ).images[0]

        generation_time = time.time() - start_time

        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"qwen_image_{timestamp}_{i+1}.png"
        image.save(filename)

        print(f"✓ Saved to {filename}")
        print(f"  Generation time: {generation_time:.2f} seconds")

    print("\n✨ Generation complete!")


if __name__ == "__main__":
    main()
