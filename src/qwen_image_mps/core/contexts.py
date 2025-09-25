from __future__ import annotations

import secrets
from enum import Enum
from typing import Callable, List, Optional, Sequence

from pydantic import BaseModel, Field, field_validator, model_validator

DEFAULT_OUTPUT_DIR = "output"
ALLOWED_ASPECTS: Sequence[str] = (
    "1:1",
    "16:9",
    "9:16",
    "4:3",
    "3:4",
    "3:2",
    "2:3",
)


class GenerationContext(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    steps: int = Field(..., ge=1)
    fast: bool = False
    ultra_fast: bool = False
    lightning_lora_filename: Optional[str] = None
    seed: Optional[int] = None
    num_images: int = Field(default=1, ge=1)
    aspect: str = Field(default="16:9")
    lora: Optional[str] = None
    quantization: Optional[str] = None
    cfg_scale: Optional[float] = None
    output_dir: Optional[str] = None
    output_filename: Optional[str] = None
    batman: bool = False
    memory_efficient: bool = False
    event_callback: Optional[Callable[[Enum], None]] = None

    @field_validator("aspect")
    @classmethod
    def validate_aspect(cls, value: str) -> str:
        if value not in ALLOWED_ASPECTS:
            raise ValueError(f"Unsupported aspect ratio: {value}")
        return value

    @field_validator("cfg_scale")
    @classmethod
    def validate_cfg_scale(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and value < 0:
            raise ValueError("cfg_scale must be non-negative")
        return value

    @model_validator(mode="after")
    def ensure_mutually_exclusive_modes(self) -> "GenerationContext":
        if self.fast and self.ultra_fast:
            raise ValueError("'fast' and 'ultra_fast' modes are mutually exclusive")
        return self

    @property
    def batman_enabled(self) -> bool:
        return bool(self.batman)

    @property
    def negative_prompt_text(self) -> str:
        return " " if self.negative_prompt is None else self.negative_prompt

    @property
    def output_directory(self) -> str:
        return self.output_dir or DEFAULT_OUTPUT_DIR

    def seed_for_image(self, index: int) -> int:
        if self.seed is not None:
            return int(self.seed) + index
        return secrets.randbits(63)

    @classmethod
    def from_args(cls, args) -> "GenerationContext":  # pragma: no cover - simple wiring
        return cls(
            prompt=args.prompt,
            negative_prompt=getattr(args, "negative_prompt", None),
            steps=int(getattr(args, "steps", 1)),
            fast=bool(getattr(args, "fast", False)),
            ultra_fast=bool(getattr(args, "ultra_fast", False)),
            lightning_lora_filename=getattr(args, "lightning_lora_filename", None),
            seed=getattr(args, "seed", None),
            num_images=int(getattr(args, "num_images", 1)),
            aspect=getattr(args, "aspect", "16:9"),
            lora=getattr(args, "lora", None),
            quantization=getattr(args, "quantization", None),
            cfg_scale=getattr(args, "cfg_scale", None),
            output_dir=getattr(args, "output_dir", None),
            output_filename=getattr(args, "output_filename", None),
            batman=bool(getattr(args, "batman", False)),
            memory_efficient=bool(getattr(args, "memory_efficient", False)),
            event_callback=getattr(args, "event_callback", None),
        )


class EditContext(BaseModel):
    input_paths: List[str]
    prompt: str
    negative_prompt: Optional[str] = None
    steps: int = Field(..., ge=1)
    fast: bool = False
    ultra_fast: bool = False
    lightning_lora_filename: Optional[str] = None
    seed: Optional[int] = None
    lora: Optional[str] = None
    quantization: Optional[str] = None
    cfg_scale: Optional[float] = None
    output_dir: Optional[str] = None
    output_filename: Optional[str] = None
    output: Optional[str] = None
    batman: bool = False
    memory_efficient: bool = False

    @field_validator("input_paths")
    @classmethod
    def validate_input_paths(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("At least one input image path is required")
        return value

    @field_validator("cfg_scale")
    @classmethod
    def validate_cfg_scale(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and value < 0:
            raise ValueError("cfg_scale must be non-negative")
        return value

    @model_validator(mode="after")
    def ensure_mutually_exclusive_modes(self) -> "EditContext":
        if self.fast and self.ultra_fast:
            raise ValueError("'fast' and 'ultra_fast' modes are mutually exclusive")
        return self

    @property
    def batman_enabled(self) -> bool:
        return bool(self.batman)

    @property
    def negative_prompt_text(self) -> str:
        return " " if self.negative_prompt is None else self.negative_prompt

    @property
    def output_directory(self) -> str:
        return self.output_dir or DEFAULT_OUTPUT_DIR

    @classmethod
    def from_args(cls, args) -> "EditContext":  # pragma: no cover - simple wiring
        raw_input = getattr(args, "input", None)
        if isinstance(raw_input, (list, tuple)):
            input_paths = list(raw_input)
        elif raw_input is None:
            input_paths = []
        else:
            input_paths = [raw_input]

        return cls(
            input_paths=input_paths,
            prompt=args.prompt,
            negative_prompt=getattr(args, "negative_prompt", None),
            steps=int(getattr(args, "steps", 1)),
            fast=bool(getattr(args, "fast", False)),
            ultra_fast=bool(getattr(args, "ultra_fast", False)),
            lightning_lora_filename=getattr(args, "lightning_lora_filename", None),
            seed=getattr(args, "seed", None),
            lora=getattr(args, "lora", None),
            quantization=getattr(args, "quantization", None),
            cfg_scale=getattr(args, "cfg_scale", None),
            output_dir=getattr(args, "output_dir", None),
            output_filename=getattr(args, "output_filename", None),
            output=getattr(args, "output", None),
            batman=bool(getattr(args, "batman", False)),
            memory_efficient=bool(getattr(args, "memory_efficient", False)),
        )

    def seed_value(self) -> int:
        if self.seed is not None:
            return int(self.seed)
        return secrets.randbits(63)
