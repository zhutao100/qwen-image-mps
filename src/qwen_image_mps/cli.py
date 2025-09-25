import argparse
import os
import secrets
from datetime import datetime
from enum import Enum

import torch

from .core.lora import (
    get_custom_lora_path,
    get_lora_path,
    merge_lora_from_safetensors,
)
from .core.pipelines import (
    get_device_and_dtype,
    load_gguf_pipeline,
)
from .prompts import (
    build_edit_prompt,
    build_generation_prompt,
    sanitize_prompt_for_filename,
)
from .utils import create_generator


class GenerationStep(Enum):
    """Enum for tracking important steps in the image generation process"""

    INIT = "init"
    LOADING_MODEL = "loading_model"
    MODEL_LOADED = "model_loaded"
    LOADING_CUSTOM_LORA = "loading_custom_lora"
    LOADING_ULTRA_FAST_LORA = "loading_ultra_fast_lora"
    LOADING_FAST_LORA = "loading_fast_lora"
    LORA_LOADED = "lora_loaded"
    BATMAN_MODE_ACTIVATED = "batman_mode_activated"
    PREPARING_GENERATION = "preparing_generation"
    INFERENCE_START = "inference_start"
    INFERENCE_PROGRESS = "inference_progress"
    INFERENCE_COMPLETE = "inference_complete"
    SAVING_IMAGE = "saving_image"
    IMAGE_SAVED = "image_saved"
    COMPLETE = "complete"
    ERROR = "error"


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
    return parser









def _emit_generation_event(event_callback, step: GenerationStep):
    if event_callback:
        try:
            event_callback(step)
        except Exception as e:
            print(f"Warning: Event callback error: {e}")
    return step


def _load_generation_pipeline(args, device, torch_dtype):
    from diffusers import DiffusionPipeline

    model_name = "Qwen/Qwen-Image"
    quantization = getattr(args, "quantization", None)

    if quantization:
        pipe = load_gguf_pipeline(quantization, device, torch_dtype, edit_mode=False)
        if pipe is None:
            print("Failed to load GGUF model, falling back to standard model...")
            pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
            pipe = pipe.to(device)
        return pipe, True

    pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
    pipe = pipe.to(device)
    return pipe, False


def _apply_custom_lora_for_generation(pipeline, args, using_gguf, emit):
    if not getattr(args, "lora", None):
        return pipeline

    if using_gguf:
        print("Warning: LoRA loading is not supported with GGUF quantized models.")
        print("The internal structure of GGUF models differs from standard models.")
        print("Continuing without LoRA...")
        return pipeline

    yield emit(GenerationStep.LOADING_CUSTOM_LORA)
    print(f"Loading custom LoRA: {args.lora}")
    custom_lora_path = get_custom_lora_path(args.lora)
    if custom_lora_path:
        pipeline = merge_lora_from_safetensors(pipeline, custom_lora_path)
        yield emit(GenerationStep.LORA_LOADED)
    else:
        print("Warning: Could not load custom LoRA, continuing without it...")
    return pipeline


def _apply_lightning_generation_mode(pipeline, args, using_gguf, emit):
    num_steps = args.steps
    cfg_scale = 4.0
    lightning_filename = getattr(args, "lightning_lora_filename", None)

    if getattr(args, "ultra_fast", False):
        if using_gguf:
            print("Warning: Lightning LoRA is not compatible with GGUF quantized models.")
            print("Using GGUF model with standard inference settings...")
            return pipeline, 4, 1.0

        yield emit(GenerationStep.LOADING_ULTRA_FAST_LORA)
        print("Loading Lightning LoRA v1.0 for ultra-fast generation...")
        lora_path = get_lora_path(
            ultra_fast=True, lightning_lora_filename=lightning_filename
        )
        if lora_path:
            pipeline = merge_lora_from_safetensors(pipeline, lora_path)
            num_steps = 4
            cfg_scale = 1.0
            yield emit(GenerationStep.LORA_LOADED)
            print(f"Ultra-fast mode enabled: {num_steps} steps, CFG scale {cfg_scale}")
        else:
            print("Warning: Could not load Lightning LoRA v1.0")
            print("Falling back to normal generation...")
    elif getattr(args, "fast", False):
        if using_gguf:
            print("Warning: Lightning LoRA is not compatible with GGUF quantized models.")
            print("Using GGUF model with reduced steps for faster generation...")
            return pipeline, 8, 1.0

        yield emit(GenerationStep.LOADING_FAST_LORA)
        print("Loading Lightning LoRA for fast generation...")
        lora_path = get_lora_path(
            ultra_fast=False, lightning_lora_filename=lightning_filename
        )
        if lora_path:
            pipeline = merge_lora_from_safetensors(pipeline, lora_path)
            num_steps = 8
            cfg_scale = 1.0
            yield emit(GenerationStep.LORA_LOADED)
            print(f"Fast mode enabled: {num_steps} steps, CFG scale {cfg_scale}")
        else:
            print("Warning: Could not load Lightning LoRA v1.1")
            print("Falling back to normal generation...")

    return pipeline, num_steps, cfg_scale


def _resolve_aspect_dimensions(aspect):
    aspect_ratios = {
        "1:1": (1328, 1328),
        "16:9": (1664, 928),
        "9:16": (928, 1664),
        "4:3": (1472, 1104),
        "3:4": (1104, 1472),
        "3:2": (1584, 1056),
        "2:3": (1056, 1584),
    }
    return aspect_ratios[aspect]


def _compute_generation_output_spec(args, timestamp):
    output_dir = getattr(args, "output_dir", None) or "output"
    if getattr(args, "output_filename", None):
        base_name = args.output_filename
    else:
        sanitized_prompt = sanitize_prompt_for_filename(args.prompt)
        base_name = f"{timestamp}-{sanitized_prompt}"
    return output_dir, base_name


def _resolve_per_image_seed(base_seed, image_index):
    if base_seed is not None:
        return int(base_seed) + image_index
    return secrets.randbits(63)



def generate_image(args):
    event_callback = getattr(args, "event_callback", None)

    def emit(step: GenerationStep):
        return _emit_generation_event(event_callback, step)

    try:
        yield emit(GenerationStep.INIT)

        device, torch_dtype = get_device_and_dtype()

        yield emit(GenerationStep.LOADING_MODEL)
        pipe, using_gguf = _load_generation_pipeline(args, device, torch_dtype)

        yield emit(GenerationStep.MODEL_LOADED)

        pipe = yield from _apply_custom_lora_for_generation(pipe, args, using_gguf, emit)

        pipe, num_steps, cfg_scale = yield from _apply_lightning_generation_mode(
            pipe, args, using_gguf, emit
        )

        if getattr(args, "cfg_scale", None) is not None:
            try:
                cfg_scale = float(args.cfg_scale)
                if cfg_scale < 0:
                    cfg_scale = 0.0
            except Exception:
                pass

        batman_enabled = getattr(args, "batman", False)
        if batman_enabled:
            yield emit(GenerationStep.BATMAN_MODE_ACTIVATED)
            print("\n🦇 BATMAN MODE ACTIVATED: Adding surprise LEGO Batman photobomb!")

        yield emit(GenerationStep.PREPARING_GENERATION)

        negative_prompt = (
            " "
            if getattr(args, "negative_prompt", None) is None
            else args.negative_prompt
        )

        width, height = _resolve_aspect_dimensions(args.aspect)
        num_images = max(1, int(args.num_images))
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        output_dir, base_name = _compute_generation_output_spec(args, timestamp)

        saved_paths = []
        saved_seeds = []

        for image_index in range(num_images):
            per_image_seed = _resolve_per_image_seed(args.seed, image_index)
            current_prompt = build_generation_prompt(
                args.prompt, batman_enabled, num_images, image_index
            )

            generator = create_generator(device, per_image_seed)

            yield emit(GenerationStep.INFERENCE_START)
            with torch.inference_mode():
                image = pipe(
                    prompt=current_prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_steps,
                    true_cfg_scale=cfg_scale,
                    generator=generator,
                ).images[0]
            yield emit(GenerationStep.INFERENCE_COMPLETE)

            yield emit(GenerationStep.SAVING_IMAGE)
            suffix = f"-{image_index + 1}" if num_images > 1 else ""
            output_filename = os.path.join(
                output_dir, f"{base_name}{suffix}.png"
            )

            os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)

            image.save(output_filename)
            abs_path = os.path.abspath(output_filename)
            saved_paths.append(abs_path)
            saved_seeds.append(per_image_seed)
            yield emit(GenerationStep.IMAGE_SAVED)

        if len(saved_paths) == 1:
            print(f"\nImage saved to: {saved_paths[0]} (seed: {saved_seeds[0]})")
        else:
            print("\nImages saved:")
            for path, seed_val in zip(saved_paths, saved_seeds):
                print(f"- {path} (seed: {seed_val})")

        yield emit(GenerationStep.COMPLETE)
        yield saved_paths

    except Exception as e:
        yield emit(GenerationStep.ERROR)
        print(f"Error during image generation: {e}")
        raise


def _get_edit_pipeline_class():
    try:
        from diffusers import QwenImageEditPlusPipeline as EditPipeline
    except ImportError:
        from diffusers import QwenImageEditPipeline as EditPipeline
    return EditPipeline


def _load_edit_pipeline(args, device, torch_dtype):
    EditPipeline = _get_edit_pipeline_class()
    quantization = getattr(args, "quantization", None)

    if quantization:
        print(f"Loading GGUF quantized model ({quantization}) for image editing...")
        pipeline = load_gguf_pipeline(quantization, device, torch_dtype, edit_mode=True)
        if pipeline is None:
            print("GGUF models for editing may not be available yet.")
            print("Falling back to standard edit model...")
            pipeline = EditPipeline.from_pretrained(
                "Qwen/Qwen-Image-Edit-2509", torch_dtype=torch_dtype
            )
    else:
        print("Loading Qwen-Image-Edit model for image editing...")
        pipeline = EditPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit-2509", torch_dtype=torch_dtype
        )

    return pipeline.to(device)


def _apply_custom_lora_if_needed(pipeline, args):
    if not getattr(args, "lora", None):
        return pipeline

    print(f"Loading custom LoRA: {args.lora}")
    custom_lora_path = get_custom_lora_path(args.lora)
    if custom_lora_path:
        return merge_lora_from_safetensors(pipeline, custom_lora_path)

    print("Warning: Could not load custom LoRA, continuing without it...")
    return pipeline


def _apply_lightning_edit_mode(pipeline, args, default_steps, default_cfg):
    num_steps = default_steps
    cfg_scale = default_cfg
    lightning_filename = getattr(args, "lightning_lora_filename", None)

    if getattr(args, "ultra_fast", False):
        print("Loading Lightning Edit LoRA v1.0 (4 steps) for ultra-fast editing...")
        lora_path = get_lora_path(
            ultra_fast=True, edit_mode=True, lightning_lora_filename=lightning_filename
        )
        if lora_path:
            pipeline = merge_lora_from_safetensors(pipeline, lora_path)
            num_steps = 4
            cfg_scale = 1.0
            print(f"Ultra-fast mode enabled: {num_steps} steps, CFG scale {cfg_scale}")
        else:
            print("Warning: Could not load Lightning Edit LoRA v1.0 (4 steps)")
            print("Falling back to normal editing...")
    elif getattr(args, "fast", False):
        print("Loading Lightning Edit LoRA v1.0 for fast editing...")
        lora_path = get_lora_path(
            edit_mode=True, lightning_lora_filename=lightning_filename
        )
        if lora_path:
            pipeline = merge_lora_from_safetensors(pipeline, lora_path)
            num_steps = 8
            cfg_scale = 1.0
            print(f"Fast edit mode enabled: {num_steps} steps, CFG scale {cfg_scale}")
        else:
            print("Warning: Could not load Lightning Edit LoRA v1.0")
            print("Falling back to normal editing...")

    return pipeline, num_steps, cfg_scale


def _load_input_images(input_paths):
    from PIL import Image

    images = []
    for path in input_paths:
        image = Image.open(path).convert("RGB")
        print(f"Loaded input image: {path} ({image.size[0]}x{image.size[1]})")
        images.append(image)
    return images


def _resolve_output_path(args):
    output_dir = getattr(args, "output_dir", None) or "output"

    if getattr(args, "output", None):
        output_path = args.output
        if os.path.basename(output_path) == output_path:
            output_path = os.path.join(output_dir, output_path)
        return output_path

    base_name = getattr(args, "output_filename", None)
    if not base_name:
        sanitized_prompt = sanitize_prompt_for_filename(args.prompt)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base_name = f"edited-{timestamp}-{sanitized_prompt}"

    return os.path.join(output_dir, f"{base_name}.png")


def edit_image(args) -> None:
    device, torch_dtype = get_device_and_dtype()

    pipeline = _load_edit_pipeline(args, device, torch_dtype)

    pipeline.set_progress_bar_config(disable=None)

    pipeline = _apply_custom_lora_if_needed(pipeline, args)

    default_steps = args.steps
    default_cfg = 4.0
    pipeline, num_steps, cfg_scale = _apply_lightning_edit_mode(
        pipeline, args, default_steps, default_cfg
    )

    # Override CFG scale if provided by user
    if getattr(args, "cfg_scale", None) is not None:
        try:
            cfg_scale = float(args.cfg_scale)
            if cfg_scale < 0:
                cfg_scale = 0.0
        except Exception:
            pass

    input_paths = args.input if isinstance(args.input, (list, tuple)) else [args.input]

    try:
        images = _load_input_images(input_paths)
    except Exception as e:
        print(f"Error loading input image: {e}")
        return

    seed = args.seed if args.seed is not None else secrets.randbits(63)
    generator = create_generator(device, seed)

    batman_enabled = getattr(args, "batman", False)
    edit_prompt = build_edit_prompt(args.prompt, batman_enabled)

    edit_negative_prompt = (
        " " if getattr(args, "negative_prompt", None) is None else args.negative_prompt
    )

    print(f"Editing image with prompt: {edit_prompt}")
    print(f"Using {num_steps} inference steps...")

    with torch.inference_mode():
        pipeline_inputs = images if len(images) > 1 else images[0]
        output = pipeline(
            image=pipeline_inputs,
            prompt=edit_prompt,
            negative_prompt=edit_negative_prompt,
            num_inference_steps=num_steps,
            generator=generator,
            guidance_scale=cfg_scale,
        )
        edited_image = output.images[0]

    output_filename = _resolve_output_path(args)
    os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)

    edited_image.save(output_filename)
    print(f"\nEdited image saved to: {os.path.abspath(output_filename)} (seed: {seed})")


def main() -> None:
    try:
        from . import __version__
    except ImportError:
        # Fallback when module is loaded without package context
        __version__ = "0.4.5"

    parser = argparse.ArgumentParser(
        description="Qwen-Image MPS - Generate and edit images with Qwen models on Apple Silicon",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"qwen-image-mps {__version__}",
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Add generate and edit subcommands
    build_generate_parser(subparsers)
    build_edit_parser(subparsers)

    import sys
    try:
        # For backward compatibility, if no subcommand is given, default to "generate"
        if len(sys.argv) > 1 and sys.argv[1] not in [
            "generate",
            "edit",
            "-h",
            "--help",
            "--version",
        ]:
            sys.argv.insert(1, "generate")

        args = parser.parse_args()

        # Handle the command
        if args.command == "generate":
            # Consume the generator to execute the image generation
            list(generate_image(args))
        elif args.command == "edit":
            edit_image(args)
        else:
            parser.print_help()

    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting...")


if __name__ == "__main__":
    main()
