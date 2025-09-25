from enum import Enum
from typing import Callable, Optional


class GenerationStep(Enum):
    """Enum for tracking important steps in the image generation process."""

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


def emit_generation_event(
    event_callback: Optional[Callable[[GenerationStep], None]],
    step: GenerationStep,
) -> GenerationStep:
    if event_callback:
        try:
            event_callback(step)
        except Exception as exc:
            print(f"Warning: Event callback error: {exc}")
    return step
