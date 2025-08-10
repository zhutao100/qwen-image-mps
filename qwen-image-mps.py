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
        default='''A coffee shop entrance features a chalkboard sign reading "Apple Silicon Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "Generated with MPS on Apple Silicon". Next to it hangs a poster showing a beautiful Italian woman, and beneath the poster is written "Just try it!". Ultra HD, 4K, cinematic composition''',
        help="Prompt text to condition the image generation.",
    )
    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=50,
        help="Number of inference steps for image generation.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # Defer heavy imports until after parsing so `--help` is fast
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

    positive_magic = {
        "en": "Ultra HD, 4K, cinematic composition.",
    }

    prompt = args.prompt
    negative_prompt = " "  # using an empty string if you do not have specific concept to remove

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

    image = pipe(
        prompt=prompt + positive_magic["en"],
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=args.steps,
        true_cfg_scale=4.0,
        generator=torch.Generator(device=generator_device).manual_seed(195),
    ).images[0]

    # Save with timestamp to avoid overwriting previous generations
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_filename = f"example-{timestamp}.png"
    image.save(output_filename)


if __name__ == "__main__":
    main()
