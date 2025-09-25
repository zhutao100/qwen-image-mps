"""Backward-compatible CLI facade for qwen-image-mps."""

from ._cli.commands.edit import edit_image
from ._cli.commands.generate import generate_image
from ._cli.events import GenerationStep
from ._cli.main import main
from ._cli.parser import build_edit_parser, build_generate_parser
from .core.lora import get_lora_path, merge_lora_from_safetensors

__all__ = [
    "GenerationStep",
    "build_generate_parser",
    "build_edit_parser",
    "generate_image",
    "edit_image",
    "get_lora_path",
    "merge_lora_from_safetensors",
    "main",
]
