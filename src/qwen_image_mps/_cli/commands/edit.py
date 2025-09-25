from __future__ import annotations

import os
from datetime import datetime

import torch

from ...core.contexts import EditContext
from ...core.lora import get_custom_lora_path, get_lora_path, merge_lora_from_safetensors
from ...core.pipelines import get_device_and_dtype, load_gguf_pipeline
from ...prompts import build_edit_prompt, sanitize_prompt_for_filename
from ...utils import create_generator


def _get_edit_pipeline_class():
    try:
        from diffusers import QwenImageEditPlusPipeline as EditPipeline
    except ImportError:
        from diffusers import QwenImageEditPipeline as EditPipeline
    return EditPipeline


def _load_edit_pipeline(context: EditContext, device, torch_dtype):
    EditPipeline = _get_edit_pipeline_class()
    quantization = context.quantization

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


def _apply_custom_lora_if_needed(pipeline, context: EditContext):
    if not context.lora:
        return pipeline

    print(f"Loading custom LoRA: {context.lora}")
    custom_lora_path = get_custom_lora_path(context.lora)
    if custom_lora_path:
        return merge_lora_from_safetensors(pipeline, custom_lora_path)

    print("Warning: Could not load custom LoRA, continuing without it...")
    return pipeline


def _apply_lightning_edit_mode(
    pipeline, context: EditContext, default_steps: int, default_cfg: float
):
    num_steps = default_steps
    cfg_scale = default_cfg
    lightning_filename = context.lightning_lora_filename

    if context.ultra_fast:
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
    elif context.fast:
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


def _resolve_output_path(context: EditContext):
    output_dir = context.output_directory

    if context.output:
        output_path = context.output
        if os.path.basename(output_path) == output_path:
            output_path = os.path.join(output_dir, output_path)
        return output_path

    base_name = context.output_filename
    if not base_name:
        sanitized_prompt = sanitize_prompt_for_filename(context.prompt)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base_name = f"edited-{timestamp}-{sanitized_prompt}"

    return os.path.join(output_dir, f"{base_name}.png")


def edit_image(args) -> None:
    context = args if isinstance(args, EditContext) else EditContext.from_args(args)
    device, torch_dtype = get_device_and_dtype()

    pipeline = _load_edit_pipeline(context, device, torch_dtype)

    pipeline.set_progress_bar_config(disable=None)

    pipeline = _apply_custom_lora_if_needed(pipeline, context)

    default_steps = context.steps
    default_cfg = 4.0
    pipeline, num_steps, cfg_scale = _apply_lightning_edit_mode(
        pipeline, context, default_steps, default_cfg
    )

    if context.cfg_scale is not None:
        cfg_scale = float(context.cfg_scale)

    try:
        images = _load_input_images(context.input_paths)
    except Exception as exc:
        print(f"Error loading input image: {exc}")
        return

    seed = context.seed_value()
    generator = create_generator(device, seed)

    edit_prompt = build_edit_prompt(context.prompt, context.batman_enabled)

    edit_negative_prompt = context.negative_prompt_text

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

    output_filename = _resolve_output_path(context)
    os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)

    edited_image.save(output_filename)
    print(f"\nEdited image saved to: {os.path.abspath(output_filename)} (seed: {seed})")
