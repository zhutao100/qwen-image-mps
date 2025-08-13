import argparse
from datetime import datetime


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
        help="Use Lightning LoRA v1.1 for fast generation (8 steps).",
    )
    parser.add_argument(
        "--ultra-fast",
        action="store_true",
        help="Use Lightning LoRA v1.0 for ultra-fast generation (4 steps).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed for reproducible generation (default: 42). If not explicitly "
            "provided and generating multiple images, a new random seed is used for each image."
        ),
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=1,
        help="Number of images to generate.",
    )
    return parser


def get_lora_path(ultra_fast=False):
    """Get the Lightning LoRA from Hugging Face cache."""
    from huggingface_hub import hf_hub_download

    if ultra_fast:
        filename = "Qwen-Image-Lightning-4steps-V1.0-bf16.safetensors"
        version = "v1.0 (4-steps)"
    else:
        filename = "Qwen-Image-Lightning-8steps-V1.1.safetensors"
        version = "v1.1 (8-steps)"

    try:
        # Force re-download by setting force_download=True to avoid cache issues
        lora_path = hf_hub_download(
            repo_id="lightx2v/Qwen-Image-Lightning",
            filename=filename,
            repo_type="model",
            force_download=True,  # Force re-download to avoid cache issues
        )
        print(f"Lightning LoRA {version} loaded from: {lora_path}")
        return lora_path
    except Exception as e:
        print(f"Failed to load Lightning LoRA {version}: {e}")
        return None


def merge_lora_from_safetensors(pipe, lora_path):
    """Merge LoRA weights from safetensors file into the pipeline's transformer."""
    import safetensors.torch
    import torch

    lora_state_dict = safetensors.torch.load_file(lora_path)

    transformer = pipe.transformer
    merged_count = 0

    for name, param in transformer.named_parameters():
        # Remove .weight suffix if present to get base parameter name
        base_name = name.replace(".weight", "") if name.endswith(".weight") else name

        # Construct LoRA keys directly (no transformer prefix needed)
        lora_down_key = f"{base_name}.lora_down.weight"
        lora_up_key = f"{base_name}.lora_up.weight"
        lora_alpha_key = f"{base_name}.alpha"

        if lora_down_key in lora_state_dict and lora_up_key in lora_state_dict:
            lora_down = lora_state_dict[lora_down_key]
            lora_up = lora_state_dict[lora_up_key]

            # Get alpha value if it exists, otherwise use rank
            if lora_alpha_key in lora_state_dict:
                lora_alpha = float(lora_state_dict[lora_alpha_key])
            else:
                lora_alpha = lora_down.shape[0]  # Use rank as default

            rank = lora_down.shape[0]
            scaling_factor = lora_alpha / rank

            # Convert to float32 for computation
            lora_up = lora_up.float()
            lora_down = lora_down.float()

            # Apply LoRA: weight = weight + scaling_factor * (up @ down)
            delta_W = scaling_factor * torch.matmul(lora_up, lora_down)
            param.data = (param.data + delta_W.to(param.device)).type_as(param.data)
            merged_count += 1

    print(f"Merged {merged_count} LoRA weights into the model")
    return pipe


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # Defer heavy imports until after parsing so `--help` is fast
    import os
    import secrets
    import sys

    import torch
    from diffusers import DiffusionPipeline

    model_name = "Qwen/Qwen-Image"

    # Select device and dtype correctly
    if torch.backends.mps.is_available():
        print("Using MPS")
        device = "mps"
        torch_dtype = torch.bfloat16  # more stable on MPS
    elif torch.cuda.is_available():
        print("Using CUDA")
        device = "cuda"
        torch_dtype = torch.bfloat16
    else:
        print("Using CPU")
        device = "cpu"
        torch_dtype = torch.float32

    pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
    pipe = pipe.to(device)

    # Apply Lightning LoRA if fast or ultra-fast mode is enabled
    if args.ultra_fast:
        print("Loading Lightning LoRA v1.0 for ultra-fast generation...")
        lora_path = get_lora_path(ultra_fast=True)
        if lora_path:
            pipe = merge_lora_from_safetensors(pipe, lora_path)
            # Use fixed 4 steps for Ultra Lightning mode
            num_steps = 4
            cfg_scale = 1.0
            print(f"Ultra-fast mode enabled: {num_steps} steps, CFG scale {cfg_scale}")
        else:
            print("Warning: Could not load Lightning LoRA v1.0")
            print("Falling back to normal generation...")
            num_steps = args.steps
            cfg_scale = 4.0
    elif args.fast:
        print("Loading Lightning LoRA v1.1 for fast generation...")
        lora_path = get_lora_path(ultra_fast=False)
        if lora_path:
            pipe = merge_lora_from_safetensors(pipe, lora_path)
            # Use fixed 8 steps for Lightning mode
            num_steps = 8
            cfg_scale = 1.0
            print(f"Fast mode enabled: {num_steps} steps, CFG scale {cfg_scale}")
        else:
            print("Warning: Could not load Lightning LoRA v1.1")
            print("Falling back to normal generation...")
            num_steps = args.steps
            cfg_scale = 4.0
    else:
        num_steps = args.steps
        cfg_scale = 4.0

    prompt = args.prompt
    negative_prompt = (
        " "  # using an empty string if you do not have specific concept to remove
    )

    aspect_ratios = {
        "1:1": (1328, 1328),
        "16:9": (1664, 928),
        "9:16": (928, 1664),
        "4:3": (1472, 1140),
        "3:4": (1140, 1472),
        "3:2": (1584, 1056),
        "2:3": (1056, 1584),
    }

    width, height = aspect_ratios["16:9"]
    generator_device = "cpu" if device == "mps" else device

    # Ensure we generate at least one image
    num_images = max(1, int(args.num_images))

    # Shared timestamp for this generation batch
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    saved_paths = []

    # Detect whether the user explicitly provided --seed on the command line
    argv = sys.argv[1:]
    seed_provided = ("--seed" in argv) or any(arg.startswith("--seed=") for arg in argv)
    for image_index in range(num_images):
        if seed_provided:
            # Deterministic: increment seed per image starting from the provided seed
            per_image_seed = int(args.seed) + image_index
        elif num_images > 1:
            # Non-deterministic for multi-image when no seed explicitly provided
            # Use 63-bit to keep it positive and well within torch's expected range
            per_image_seed = secrets.randbits(63)
        else:
            # Single image without explicit seed: use default (42)
            per_image_seed = int(args.seed)
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_steps,
            true_cfg_scale=cfg_scale,
            generator=torch.Generator(device=generator_device).manual_seed(
                per_image_seed
            ),
        ).images[0]

        # Save with timestamp to avoid overwriting previous generations
        suffix = f"-{image_index+1}" if num_images > 1 else ""
        output_filename = f"image-{timestamp}{suffix}.png"
        image.save(output_filename)
        saved_paths.append(os.path.abspath(output_filename))

    # Print full path(s) of saved image(s)
    if len(saved_paths) == 1:
        print(f"\nImage saved to: {saved_paths[0]}")
    else:
        print("\nImages saved:")
        for path in saved_paths:
            print(f"- {path}")


if __name__ == "__main__":
    main()
