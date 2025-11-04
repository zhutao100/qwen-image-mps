import argparse


def build_generate_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "generate",
        help="Generate a new image from text prompt",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default="""A coffee shop entrance features a chalkboard sign reading "Apple Silicon Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "Generated with MPS on Apple Silicon". Next to it hangs a poster showing a beautiful woman, and beneath the poster is written "Just try it!".""",
        help="Prompt text to condition the image generation.",
    )
    parser.add_argument(
        "-np",
        "--negative-prompt",
        dest="negative_prompt",
        type=str,
        default=None,
        help=(
            "Text to discourage (negative prompt), e.g. 'blurry, watermark, text, low quality'. "
            "If omitted, an empty negative prompt is used."
        ),
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
        help="Use Lightning LoRA for fast generation (8 steps).",
    )
    parser.add_argument(
        "-uf",
        "--ultra-fast",
        action="store_true",
        help="Use Lightning LoRA for ultra-fast generation (4 steps).",
    )
    parser.add_argument(
        "--lightning-lora-filename",
        type=str,
        default=None,
        help="Filename of the Lightning LoRA to use (e.g., 'Qwen-Image-Lightning-8steps-V1.1.safetensors'). If not provided, the default filename based on mode will be used.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Random seed for reproducible generation. If not provided, a random seed "
            "will be used for each image."
        ),
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=1,
        help="Number of images to generate.",
    )
    parser.add_argument(
        "--aspect",
        type=str,
        choices=["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"],
        default="16:9",
        help="Aspect ratio for the output image.",
    )
    parser.add_argument(
        "--lora",
        type=str,
        default=None,
        help="Path to local .safetensors file, Hugging Face model URL or repo ID for additional LoRA to load (e.g., '~/Downloads/lora.safetensors', 'flymy-ai/qwen-image-anime-irl-lora' or full HF URL).",
    )
    parser.add_argument(
        "--cfg-scale",
        dest="cfg_scale",
        type=float,
        default=None,
        help="Classifier-free guidance (CFG) scale. Overrides mode defaults (fast/ultra-fast use 1.0; normal uses 4.0).",
    )
    parser.add_argument(
        "--outdir",
        dest="output_dir",
        type=str,
        default=None,
        help="Directory to save generated images (default: ./output).",
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default=None,
        help="Base name for the output files (without extension). If not provided, a timestamp-based name will be used.",
    )
    parser.add_argument(
        "--dimensions",
        type=str,
        default=None,
        help="Optional output size, e.g. '1920x1080', '1920,1080', or '[1920 1080]'.",
    )
    parser.add_argument(
        "--batman",
        action="store_true",
        help="LEGO Batman photobombs your image! 🦇",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=[
            "Q2_K",
            "Q3_K_S",
            "Q3_K_M",
            "Q4_0",
            "Q4_1",
            "Q4_K_S",
            "Q4_K_M",
            "Q5_0",
            "Q5_1",
            "Q5_K_S",
            "Q5_K_M",
            "Q6_K",
            "Q8_0",
        ],
        help="Use GGUF quantized model (e.g., Q4_0 for ~3x memory reduction). Default uses standard model.",
    )
    parser.add_argument(
        "--memory-efficient",
        action="store_true",
        help="Enable memory-efficient optimizations.",
    )
    return parser


def build_edit_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "edit",
        help="Edit an existing image using text instructions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to the input image(s) to edit. Provide multiple paths to blend edits across images.",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        required=True,
        help="Editing instructions (e.g., 'Change the sky to sunset colors').",
    )
    parser.add_argument(
        "-np",
        "--negative-prompt",
        dest="negative_prompt",
        type=str,
        default=None,
        help=(
            "Text to discourage in the edit (negative prompt). If omitted, an empty negative prompt is used."
        ),
    )
    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=40,
        help="Number of inference steps for normal editing.",
    )
    parser.add_argument(
        "-f",
        "--fast",
        action="store_true",
        help="Use Lightning LoRA for fast editing (8 steps).",
    )
    parser.add_argument(
        "-uf",
        "--ultra-fast",
        action="store_true",
        help="Use Lightning LoRA for ultra-fast editing (4 steps).",
    )
    parser.add_argument(
        "--lightning-lora-filename",
        type=str,
        default=None,
        help="Filename of the Lightning LoRA to use (e.g., 'Qwen-Image-Lightning-8steps-V1.1.safetensors'). If not provided, the default filename based on mode will be used.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible generation. If not provided, a random seed will be used.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output filename (default: edited-<timestamp>.png).",
    )
    parser.add_argument(
        "--cfg-scale",
        dest="cfg_scale",
        type=float,
        default=None,
        help="Classifier-free guidance (CFG) scale. Overrides mode defaults (fast/ultra-fast use 1.0; normal uses 4.0).",
    )
    parser.add_argument(
        "--outdir",
        dest="output_dir",
        type=str,
        default=None,
        help="Directory to save edited images (default: ./output).",
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default=None,
        help="Base name for the output files (without extension). If not provided, a timestamp-based name will be used.",
    )
    parser.add_argument(
        "--dimensions",
        type=str,
        default=None,
        help="Optional output size, e.g. '1920x1080', '1920,1080', or '[1920 1080]'.",
    )
    parser.add_argument(
        "--lora",
        type=str,
        default=None,
        help="Path to local .safetensors file, Hugging Face model URL or repo ID for additional LoRA to load (e.g., '~/Downloads/lora.safetensors', 'flymy-ai/qwen-image-anime-irl-lora' or full HF URL).",
    )
    parser.add_argument(
        "--batman",
        action="store_true",
        help="LEGO Batman photobombs your image! 🦇",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=[
            "Q2_K",
            "Q3_K_S",
            "Q3_K_M",
            "Q4_0",
            "Q4_1",
            "Q4_K_S",
            "Q4_K_M",
            "Q5_0",
            "Q5_1",
            "Q5_K_S",
            "Q5_K_M",
            "Q6_K",
            "Q8_0",
        ],
        help="Use GGUF quantized model (e.g., Q4_0 for ~3x memory reduction). Default uses standard model.",
    )
    parser.add_argument(
        "--memory-efficient",
        action="store_true",
        help="Enable memory-efficient optimizations.",
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen-Image MPS - Generate and edit images with Qwen models on Apple Silicon",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    from .. import __version__  # type: ignore

    parser.add_argument(
        "--version",
        action="version",
        version=f"qwen-image-mps {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    build_generate_parser(subparsers)
    build_edit_parser(subparsers)
    return parser
