from __future__ import annotations

import os
from datetime import datetime
from typing import Generator

import torch

from ...core.contexts import GenerationContext
from ...core.lora import get_custom_lora_path, get_lora_path, merge_lora_from_safetensors
from ...core.pipelines import get_device_and_dtype, load_gguf_pipeline
from ...prompts import build_generation_prompt, sanitize_prompt_for_filename
from ...utils import create_generator
from ..events import GenerationStep, emit_generation_event


def _load_generation_pipeline(context: GenerationContext, device, torch_dtype):
    from diffusers import DiffusionPipeline

    model_name = "Qwen/Qwen-Image"
    quantization = context.quantization

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


def _apply_custom_lora_for_generation(pipeline, context: GenerationContext, using_gguf, emit):
    if not context.lora:
        return pipeline

    if using_gguf:
        print("Warning: LoRA loading is not supported with GGUF quantized models.")
        print("The internal structure of GGUF models differs from standard models.")
        print("Continuing without LoRA...")
        return pipeline

    yield emit(GenerationStep.LOADING_CUSTOM_LORA)
    print(f"Loading custom LoRA: {context.lora}")
    custom_lora_path = get_custom_lora_path(context.lora)
    if custom_lora_path:
        pipeline = merge_lora_from_safetensors(pipeline, custom_lora_path)
        yield emit(GenerationStep.LORA_LOADED)
    else:
        print("Warning: Could not load custom LoRA, continuing without it...")
    return pipeline


def _apply_lightning_generation_mode(
    pipeline, context: GenerationContext, using_gguf, emit
):
    num_steps = context.steps
    cfg_scale = 4.0
    lightning_filename = context.lightning_lora_filename

    if context.ultra_fast:
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
    elif context.fast:
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


def _compute_generation_output_spec(context: GenerationContext, timestamp: str):
    output_dir = context.output_directory
    if context.output_filename:
        base_name = context.output_filename
    else:
        sanitized_prompt = sanitize_prompt_for_filename(context.prompt)
        base_name = f"{timestamp}-{sanitized_prompt}"
    return output_dir, base_name


def _resolve_per_image_seed(context: GenerationContext, image_index: int) -> int:
    return context.seed_for_image(image_index)


def generate_image(args) -> Generator[GenerationStep | list[str], None, None]:
    context = (
        args if isinstance(args, GenerationContext) else GenerationContext.from_args(args)
    )
    event_callback = context.event_callback

    def emit(step: GenerationStep):
        return emit_generation_event(event_callback, step)

    try:
        yield emit(GenerationStep.INIT)

        device, torch_dtype = get_device_and_dtype()

        yield emit(GenerationStep.LOADING_MODEL)
        pipe, using_gguf = _load_generation_pipeline(context, device, torch_dtype)

        yield emit(GenerationStep.MODEL_LOADED)

        pipe = yield from _apply_custom_lora_for_generation(
            pipe, context, using_gguf, emit
        )

        pipe, num_steps, cfg_scale = yield from _apply_lightning_generation_mode(
            pipe, context, using_gguf, emit
        )

        if context.cfg_scale is not None:
            cfg_scale = float(context.cfg_scale)

        batman_enabled = context.batman_enabled
        if batman_enabled:
            yield emit(GenerationStep.BATMAN_MODE_ACTIVATED)
            print("\n🦇 BATMAN MODE ACTIVATED: Adding surprise LEGO Batman photobomb!")

        yield emit(GenerationStep.PREPARING_GENERATION)

        negative_prompt = context.negative_prompt_text

        if context.width is not None and context.height is not None:
            width, height = context.width, context.height
        else:
            width, height = _resolve_aspect_dimensions(context.aspect)

        num_images = context.num_images
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        output_dir, base_name = _compute_generation_output_spec(context, timestamp)

        saved_paths: list[str] = []
        saved_seeds: list[int] = []

        for image_index in range(num_images):
            per_image_seed = _resolve_per_image_seed(context, image_index)
            current_prompt = build_generation_prompt(
                context.prompt, batman_enabled, num_images, image_index
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

    except Exception as exc:
        yield emit(GenerationStep.ERROR)
        print(f"Error during image generation: {exc}")
        raise
