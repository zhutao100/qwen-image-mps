"""Qwen-Image MPS - Generate images with Qwen-Image on Apple Silicon and other devices."""

__version__ = "0.1.0"

from .cli import main, build_arg_parser, download_lora_if_needed, merge_lora_from_safetensors

__all__ = ["main", "build_arg_parser", "download_lora_if_needed", "merge_lora_from_safetensors"]