import os
import re
from pathlib import Path
from typing import Optional

import torch

from .pipelines import maybe_empty_mps_cache

DEFAULT_LORA_CHUNK_BYTES = 128 * 1024 * 1024
try:
    MAX_LORA_CHUNK_BYTES = int(
        os.environ.get("QWEN_MAX_LORA_CHUNK_BYTES", DEFAULT_LORA_CHUNK_BYTES)
    )
except ValueError:
    MAX_LORA_CHUNK_BYTES = DEFAULT_LORA_CHUNK_BYTES


def _dtype_size_in_bytes(dtype: torch.dtype) -> int:
    try:
        return torch.finfo(dtype).bits // 8
    except TypeError:
        return torch.tensor([], dtype=dtype).element_size()


def get_lora_path(
    ultra_fast: bool = False,
    edit_mode: bool = False,
    lightning_lora_filename: Optional[str] = None,
) -> Optional[str]:
    from huggingface_hub import hf_hub_download

    """Get the Lightning LoRA from Hugging Face Hub with a silent cache freshness check."""

    if lightning_lora_filename:
        filename = lightning_lora_filename
        version = f"custom ({lightning_lora_filename})"
    elif edit_mode and ultra_fast:
        filename = "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors"
        version = "Edit v1.0 (4-steps)"
    elif edit_mode:
        filename = "Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors"
        version = "Edit v1.0 (8-steps)"
    elif ultra_fast:
        filename = "Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors"
        version = "v2.0 (4-steps)"
    else:
        filename = "Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors"
        version = "v2.0 (8-steps)"

    try:
        latest_path = hf_hub_download(
            repo_id="lightx2v/Qwen-Image-Lightning",
            filename=filename,
            subfolder='Qwen-Image-Edit-2509' if edit_mode else None,
            repo_type="model",
        )
        print(f"Lightning LoRA {version} loaded from: {latest_path}")
        return latest_path
    except Exception as exc:
        print(f"Failed to load Lightning LoRA {version}: {exc}")
        return None


def get_custom_lora_path(lora_spec: str) -> Optional[str]:
    from huggingface_hub import hf_hub_download, list_repo_files

    lora_path = Path(lora_spec).expanduser()
    if lora_path.exists() and lora_path.suffix == ".safetensors":
        print(f"Using local LoRA file: {lora_path}")
        return str(lora_path.absolute())

    if lora_spec.startswith("https://huggingface.co/"):
        match = re.match(r"https://huggingface\\.co/([^/]+/[^/]+)", lora_spec)
        if match:
            repo_id = match.group(1)
        else:
            print(f"Invalid Hugging Face URL format: {lora_spec}")
            return None
    else:
        repo_id = lora_spec

    try:
        print(f"Looking for LoRA files in {repo_id}...")
        files = list_repo_files(repo_id, repo_type="model")
        safetensors_files = [file for file in files if file.endswith(".safetensors")]

        if not safetensors_files:
            print(f"No safetensors files found in {repo_id}")
            return None

        lora_files = [file for file in safetensors_files if "lora" in file.lower()]
        filename = lora_files[0] if lora_files else safetensors_files[0]

        print(f"Downloading LoRA file: {filename}")
        lora_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
        )

        print(f"Custom LoRA loaded from: {lora_path}")
        return lora_path
    except Exception as exc:
        print(f"Failed to load custom LoRA from {repo_id}: {exc}")
        return None


def _lora_scaling_factor(lora_down, lora_up, lora_alpha) -> float:
    rank = lora_down.shape[0] or 1
    return float(lora_alpha) / float(rank)


def _chunked_add_lora_delta(param, lora_up, lora_down, scaling_factor: float):
    target_device = param.data.device
    target_dtype = param.data.dtype

    lora_up = lora_up.to(device=target_device, dtype=target_dtype, copy=True)
    lora_down = lora_down.to(device=target_device, dtype=target_dtype, copy=True)

    out_dim = lora_up.shape[0]
    in_dim = lora_down.shape[1]

    bytes_per_elem = _dtype_size_in_bytes(target_dtype)

    full_bytes = out_dim * in_dim * bytes_per_elem

    if full_bytes <= MAX_LORA_CHUNK_BYTES:
        delta = torch.matmul(lora_up, lora_down)
        delta.mul_(scaling_factor)
        param.data.add_(delta)
        return

    rows_per_chunk = max(1, MAX_LORA_CHUNK_BYTES // max(in_dim * bytes_per_elem, 1))
    for start in range(0, out_dim, rows_per_chunk):
        end = min(start + rows_per_chunk, out_dim)
        delta_chunk = torch.matmul(lora_up[start:end], lora_down)
        delta_chunk.mul_(scaling_factor)
        param.data[start:end].add_(delta_chunk)


@torch.inference_mode()
def merge_lora_from_safetensors(pipe, lora_path: str):
    import safetensors.torch

    lora_state_dict = safetensors.torch.load_file(lora_path)

    transformer = pipe.transformer
    merged_count = 0

    lora_keys = list(lora_state_dict.keys())

    uses_dot_lora_format = any(
        ".lora.down" in key or ".lora.up" in key for key in lora_keys
    )
    uses_diffusers_format = any(key.startswith("lora_unet_") for key in lora_keys)
    uses_lora_ab_format = any(".lora_A" in key or ".lora_B" in key for key in lora_keys)

    def convert_diffusers_key_to_transformer_key(diffusers_key: str) -> str:
        key = diffusers_key.replace("lora_unet_", "")

        key = re.sub(r"transformer_blocks_(\\d+)", r"transformer_blocks.\\1", key)

        replacements = {
            "_attn_add_k_proj": ".attn.add_k_proj",
            "_attn_add_q_proj": ".attn.add_q_proj",
            "_attn_add_v_proj": ".attn.add_v_proj",
            "_attn_to_add_out": ".attn.to_add_out",
            "_ff_context_mlp_fc1": ".ff_context.net.0",
            "_ff_context_mlp_fc2": ".ff_context.net.2",
            "_ff_mlp_fc1": ".ff.net.0",
            "_ff_mlp_fc2": ".ff.net.2",
            "_attn_to_k": ".attn.to_k",
            "_attn_to_q": ".attn.to_q",
            "_attn_to_v": ".attn.to_v",
            "_attn_to_out_0": ".attn.to_out.0",
        }

        for old, new in replacements.items():
            key = key.replace(old, new)

        return key

    if uses_lora_ab_format:
        for name, param in transformer.named_parameters():
            base_name = (
                name.replace(".weight", "") if name.endswith(".weight") else name
            )

            lora_a_key = None
            lora_b_key = None

            test_key_a = f"diffusion_model.{base_name}.lora_A.weight"
            test_key_b = f"diffusion_model.{base_name}.lora_B.weight"

            if test_key_a in lora_state_dict and test_key_b in lora_state_dict:
                lora_a_key = test_key_a
                lora_b_key = test_key_b
            else:
                test_key_a = f"{base_name}.lora_A.weight"
                test_key_b = f"{base_name}.lora_B.weight"

                if test_key_a in lora_state_dict and test_key_b in lora_state_dict:
                    lora_a_key = test_key_a
                    lora_b_key = test_key_b

            if lora_a_key and lora_b_key:
                lora_down = lora_state_dict[lora_a_key]
                lora_up = lora_state_dict[lora_b_key]

                lora_alpha = lora_down.shape[0]
                scaling_factor = _lora_scaling_factor(lora_down, lora_up, lora_alpha)
                _chunked_add_lora_delta(param, lora_up, lora_down, scaling_factor)
                del lora_down, lora_up
                merged_count += 1
    elif uses_diffusers_format:
        for name, param in transformer.named_parameters():
            base_name = (
                name.replace(".weight", "") if name.endswith(".weight") else name
            )

            lora_down_key = None
            lora_up_key = None
            lora_alpha_key = None

            for key in lora_keys:
                if key.startswith("lora_unet_"):
                    converted_key = convert_diffusers_key_to_transformer_key(
                        key.replace(".lora_down.weight", "")
                        .replace(".lora_up.weight", "")
                        .replace(".alpha", "")
                    )
                    if converted_key == base_name:
                        if key.endswith(".lora_down.weight"):
                            lora_down_key = key
                        elif key.endswith(".lora_up.weight"):
                            lora_up_key = key
                        elif key.endswith(".alpha"):
                            lora_alpha_key = key

            if lora_down_key and lora_up_key:
                lora_down = lora_state_dict[lora_down_key]
                lora_up = lora_state_dict[lora_up_key]

                if lora_alpha_key and lora_alpha_key in lora_state_dict:
                    lora_alpha = float(lora_state_dict[lora_alpha_key])
                else:
                    lora_alpha = lora_down.shape[0]

                scaling_factor = _lora_scaling_factor(lora_down, lora_up, lora_alpha)
                _chunked_add_lora_delta(param, lora_up, lora_down, scaling_factor)
                del lora_down, lora_up
                merged_count += 1
    else:
        for name, param in transformer.named_parameters():
            base_name = (
                name.replace(".weight", "") if name.endswith(".weight") else name
            )

            if uses_dot_lora_format:
                lora_down_key = f"transformer.{base_name}.lora.down.weight"
                lora_up_key = f"transformer.{base_name}.lora.up.weight"
                lora_alpha_key = f"transformer.{base_name}.alpha"

                if lora_down_key not in lora_state_dict:
                    lora_down_key = f"{base_name}.lora.down.weight"
                    lora_up_key = f"{base_name}.lora.up.weight"
                    lora_alpha_key = f"{base_name}.alpha"
            else:
                lora_down_key = f"{base_name}.lora_down.weight"
                lora_up_key = f"{base_name}.lora_up.weight"
                lora_alpha_key = f"{base_name}.alpha"

            if lora_down_key in lora_state_dict and lora_up_key in lora_state_dict:
                lora_down = lora_state_dict[lora_down_key]
                lora_up = lora_state_dict[lora_up_key]

                if lora_alpha_key in lora_state_dict:
                    lora_alpha = float(lora_state_dict[lora_alpha_key])
                else:
                    lora_alpha = lora_down.shape[0]

                scaling_factor = _lora_scaling_factor(lora_down, lora_up, lora_alpha)
                _chunked_add_lora_delta(param, lora_up, lora_down, scaling_factor)
                del lora_down, lora_up
                merged_count += 1

    print(f"Merged {merged_count} LoRA weights into the model")
    del lora_state_dict
    maybe_empty_mps_cache()
    return pipe
