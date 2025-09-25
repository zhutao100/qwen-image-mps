import random
import re

BATMAN_GENERATION_VARIANTS = [
    ", with a tiny LEGO Batman minifigure photobombing in the corner doing a dramatic cape pose",
    ", featuring a small LEGO Batman minifigure sneaking into the frame from the side",
    ", and a miniature LEGO Batman figure peeking from behind something",
    ", with a tiny LEGO Batman minifigure in the background striking a heroic pose",
    ", including a small LEGO Batman figure hanging upside down from the top of the frame",
    ", with a tiny LEGO Batman minifigure doing the Batusi dance in the corner",
    ", and a small LEGO Batman figure photobombing with jazz hands",
    ", featuring a miniature LEGO Batman popping up from the bottom like 'I'm Batman!'",
    ", with a tiny LEGO Batman minifigure sliding into frame on a grappling hook",
    ", and a small LEGO Batman figure in the distance shouting 'WHERE ARE THEY?!'",
]

BATMAN_EDIT_VARIANTS = [
    " Also add a tiny LEGO Batman minifigure photobombing somewhere unexpected.",
    " Include a small LEGO Batman figure sneaking into the scene.",
    " Add a miniature LEGO Batman peeking from an edge.",
    " Put a tiny LEGO Batman minifigure doing something heroic in the background.",
    " Add a small LEGO Batman figure photobombing with a dramatic pose.",
    " Include a tiny LEGO Batman minifigure who looks like he's saying 'I'm Batman!'",
    " Add a miniature LEGO Batman swinging on a tiny grappling hook.",
    " Include a small LEGO Batman figure doing the Batusi dance.",
    " Add a tiny LEGO Batman minifigure brooding mysteriously in a corner.",
    " Put a small LEGO Batman photobombing like he's protecting Gotham.",
]


def sanitize_prompt_for_filename(prompt: str, max_length: int = 50) -> str:
    sanitized = "".join(c for c in prompt if c.isalnum() or c in " .-_").strip()
    sanitized = re.sub(r"\s+", " ", sanitized)
    return sanitized[:max_length]


def build_generation_prompt(
    base_prompt: str,
    batman_enabled: bool,
    num_images: int,
    image_index: int,
) -> str:
    if not batman_enabled:
        return base_prompt

    batman_action = random.choice(BATMAN_GENERATION_VARIANTS)
    if num_images > 1:
        print(
            f"  Image {image_index + 1}: Using Batman variant - {batman_action[2:50]}..."
        )
    return base_prompt + batman_action


def build_edit_prompt(base_prompt: str, batman_enabled: bool) -> str:
    if not batman_enabled:
        return base_prompt

    batman_edit = random.choice(BATMAN_EDIT_VARIANTS)
    print("\n🦇 BATMAN MODE ACTIVATED: LEGO Batman will photobomb this edit!")
    return base_prompt + batman_edit
