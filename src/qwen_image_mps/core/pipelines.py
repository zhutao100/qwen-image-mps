from __future__ import annotations

from typing import Optional

import torch


def get_device_and_dtype():
    """Get the optimal device and dtype for the current system."""

    if torch.backends.mps.is_available():
        print("Using MPS")
        return "mps", torch.bfloat16
    if torch.cuda.is_available():
        print("Using CUDA")
        return "cuda", torch.bfloat16
    print("Using CPU")
    return "cpu", torch.float32


def get_gguf_model_path(quantization: str) -> Optional[str]:
    from huggingface_hub import hf_hub_download

    gguf_files = {
        "Q2_K": "qwen-image-Q2_K.gguf",
        "Q3_K_S": "qwen-image-Q3_K_S.gguf",
        "Q3_K_M": "qwen-image-Q3_K_M.gguf",
        "Q4_0": "qwen-image-Q4_0.gguf",
        "Q4_1": "qwen-image-Q4_1.gguf",
        "Q4_K_S": "qwen-image-Q4_K_S.gguf",
        "Q4_K_M": "qwen-image-Q4_K_M.gguf",
        "Q5_0": "qwen-image-Q5_0.gguf",
        "Q5_1": "qwen-image-Q5_1.gguf",
        "Q5_K_S": "qwen-image-Q5_K_S.gguf",
        "Q5_K_M": "qwen-image-Q5_K_M.gguf",
        "Q6_K": "qwen-image-Q6_K.gguf",
        "Q8_0": "qwen-image-Q8_0.gguf",
    }

    if quantization not in gguf_files:
        raise ValueError(f"Unsupported quantization level: {quantization}")

    filename = gguf_files[quantization]
    print(f"Downloading GGUF model with {quantization} quantization...")

    try:
        gguf_path = hf_hub_download(
            repo_id="city96/Qwen-Image-gguf",
            filename=filename,
            repo_type="model",
        )
        print(f"GGUF model downloaded: {gguf_path}")
        return gguf_path
    except Exception as exc:
        print(f"Failed to download GGUF model: {exc}")
        return None


def load_gguf_pipeline(
    quantization: str,
    device: str,
    torch_dtype,
    edit_mode: bool = False,
):
    try:
        from diffusers import GGUFQuantizationConfig
    except ImportError:
        print(
            "Error: GGUFQuantizationConfig not found. Please update diffusers to version >=0.35.0"
        )
        return None

    gguf_path = get_gguf_model_path(quantization)
    if not gguf_path:
        return None

    quantization_config = GGUFQuantizationConfig(compute_dtype=torch_dtype)

    if edit_mode:
        print("Note: GGUF quantized models for editing are not yet supported.")
        print("The GGUF models from city96 are for the base Qwen-Image model only.")
        return None

    from diffusers import DiffusionPipeline

    print(f"Loading GGUF quantized model ({quantization})...")
    mem = get_total_memory_estimate(quantization)
    if mem is not None:
        print(
            "Estimated on-device memory (transformer + text encoder + VAE): "
            f"{mem['formatted']}"
        )
    else:
        print(
            f"Estimated transformer memory: {get_model_size(quantization)}; "
            "text encoder (~16.6 GB) and VAE (~1.2 GB) additionally."
        )

    try:
        from diffusers.models import QwenImageTransformer2DModel

        print("Loading transformer from GGUF file...")
        try:
            transformer = QwenImageTransformer2DModel.from_single_file(
                gguf_path,
                quantization_config=quantization_config,
                torch_dtype=torch_dtype,
                config="Qwen/Qwen-Image",
                subfolder="transformer",
            )
        except Exception as first_exc:
            print(f"First attempt failed: {first_exc}")
            try:
                transformer = QwenImageTransformer2DModel.from_single_file(
                    gguf_path,
                    torch_dtype=torch_dtype,
                    config="Qwen/Qwen-Image",
                    subfolder="transformer",
                )
            except Exception as second_exc:
                print(f"Second attempt failed: {second_exc}")
                transformer = QwenImageTransformer2DModel.from_single_file(
                    gguf_path,
                    torch_dtype=torch_dtype,
                )

        print("Creating pipeline with quantized transformer...")

        pipeline = DiffusionPipeline.from_pretrained(
            "Qwen/Qwen-Image",
            transformer=transformer,
            torch_dtype=torch_dtype,
        )

        pipeline = pipeline.to(device)
        print(f"Successfully loaded GGUF model with {quantization} quantization")
        return pipeline

    except (ImportError, AttributeError) as exc:
        print(f"Error: Could not import QwenImageTransformer2DModel: {exc}")
        print(
            "This might be because your diffusers version doesn't support GGUF for Qwen-Image yet."
        )
        print("Falling back to standard transformer...")

        pipeline = DiffusionPipeline.from_pretrained(
            "Qwen/Qwen-Image",
            torch_dtype=torch_dtype,
        )
        pipeline = pipeline.to(device)
        print("Successfully loaded standard model")
        return pipeline

    except Exception as exc:
        print(f"Error loading GGUF model: {exc}")
        print(
            "The GGUF file format might not be compatible with the current diffusers implementation."
        )
        print("Falling back to standard transformer...")

        pipeline = DiffusionPipeline.from_pretrained(
            "Qwen/Qwen-Image",
            torch_dtype=torch_dtype,
        )
        pipeline = pipeline.to(device)
        print("Successfully loaded standard model")
        return pipeline


def load_quantized_text_encoder(quantization: str, device: str, torch_dtype):
    try:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        if quantization == "4bit":
            _ = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
        elif quantization == "8bit":
            _ = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            print(f"Unsupported quantization type: {quantization}")
            return None

        print(f"Loading text encoder with {quantization} quantization...")
        return None

    except ImportError:
        print("bitsandbytes is required for text encoder quantization")
        print("Install with: pip install bitsandbytes")
        return None


def get_text_encoder_gguf_path(quantization: str) -> Optional[str]:
    from huggingface_hub import hf_hub_download

    gguf_files = {
        "Q2_K": "Qwen2.5-VL-7B-Instruct-Q2_K.gguf",
        "Q3_K_S": "Qwen2.5-VL-7B-Instruct-Q3_K_S.gguf",
        "Q3_K_M": "Qwen2.5-VL-7B-Instruct-Q3_K_M.gguf",
        "Q4_0": "Qwen2.5-VL-7B-Instruct-Q4_0.gguf",
        "Q4_1": "Qwen2.5-VL-7B-Instruct-Q4_1.gguf",
        "Q4_K_S": "Qwen2.5-VL-7B-Instruct-Q4_K_S.gguf",
        "Q4_K_M": "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
        "Q5_0": "Qwen2.5-VL-7B-Instruct-Q5_0.gguf",
        "Q5_1": "Qwen2.5-VL-7B-Instruct-Q5_1.gguf",
        "Q5_K_S": "Qwen2.5-VL-7B-Instruct-Q5_K_S.gguf",
        "Q5_K_M": "Qwen2.5-VL-7B-Instruct-Q5_K_M.gguf",
        "Q6_K": "Qwen2.5-VL-7B-Instruct-Q6_K.gguf",
        "Q8_0": "Qwen2.5-VL-7B-Instruct-Q8_0.gguf",
    }

    if quantization not in gguf_files:
        print(f"Unsupported text encoder quantization level: {quantization}")
        return None

    filename = gguf_files[quantization]
    print(f"Downloading GGUF text encoder with {quantization} quantization...")

    try:
        gguf_path = hf_hub_download(
            repo_id="unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
            filename=filename,
            repo_type="model",
        )
        print(f"GGUF text encoder downloaded: {gguf_path}")
        return gguf_path
    except Exception as exc:
        print(f"Error downloading GGUF text encoder: {exc}")
        return None


class QwenTextEncoderGGUF:
    def __init__(self, gguf_path: str, device: str, torch_dtype):
        self.device = device
        self.torch_dtype = torch_dtype
        self.gguf_path = gguf_path
        self._load_model()

    def _load_model(self):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"Loading GGUF text encoder from {self.gguf_path}...")

            self.model = AutoModelForCausalLM.from_pretrained(
                self.gguf_path,
                torch_dtype=self.torch_dtype,
                device_map="auto" if self.device != "cpu" else None,
                trust_remote_code=True,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True
            )

            if self.device == "cpu":
                self.model = self.model.to(self.device)

            print("GGUF text encoder loaded successfully")

        except Exception as exc:
            print(f"Error loading GGUF text encoder: {exc}")
            raise

    def encode(self, text, return_tensors: str = "pt"):
        try:
            inputs = self.tokenizer(
                text,
                return_tensors=return_tensors,
                padding=True,
                truncation=True,
                max_length=512,
            )

            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                embeddings = outputs.hidden_states[-1]

            return embeddings

        except Exception as exc:
            print(f"Error encoding text: {exc}")
            raise

    def __call__(self, text, return_tensors: str = "pt"):
        return self.encode(text, return_tensors)


def load_gguf_text_encoder(quantization: str, device: str, torch_dtype):
    try:
        gguf_path = get_text_encoder_gguf_path(quantization)
        if not gguf_path:
            return None

        text_encoder = QwenTextEncoderGGUF(gguf_path, device, torch_dtype)
        return text_encoder

    except Exception as exc:
        print(f"Error loading GGUF text encoder: {exc}")
        return None


def get_model_size(quantization: str) -> str:
    sizes = {
        "Q2_K": "7.06 GB",
        "Q3_K_S": "8.95 GB",
        "Q3_K_M": "9.68 GB",
        "Q4_0": "11.9 GB",
        "Q4_1": "12.8 GB",
        "Q4_K_S": "12.1 GB",
        "Q4_K_M": "13.1 GB",
        "Q5_0": "14.4 GB",
        "Q5_1": "15.4 GB",
        "Q5_K_S": "14.1 GB",
        "Q5_K_M": "14.9 GB",
        "Q6_K": "16.8 GB",
        "Q8_0": "21.8 GB",
    }
    return sizes.get(quantization, "Unknown")


def get_text_encoder_size(quantization: str) -> str:
    return "16.6 GB"


def get_total_memory_estimate(quantization: str):
    transformer_str = get_model_size(quantization)
    text_encoder_str = get_text_encoder_size(quantization)

    if transformer_str == "Unknown":
        return None

    try:
        transformer_gb = float(transformer_str.replace(" GB", "").strip())
        text_encoder_gb = float(text_encoder_str.replace(" GB", "").strip())
    except Exception:
        return None

    vae_and_overhead_gb = 1.2
    total_gb = round(transformer_gb + text_encoder_gb + vae_and_overhead_gb, 1)

    return {
        "transformer_gb": transformer_gb,
        "text_encoder_gb": text_encoder_gb,
        "vae_gb": vae_and_overhead_gb,
        "total_gb": total_gb,
        "breakdown": (
            f"~{transformer_gb} (transformer) + ~{text_encoder_gb} (text encoder) + "
            f"~{vae_and_overhead_gb} (VAE/overhead)"
        ),
        "formatted": (
            f"~{total_gb} GB (transformer ~{transformer_gb} GB + "
            f"text encoder ~{text_encoder_gb} GB + "
            f"VAE/overhead ~{vae_and_overhead_gb} GB)"
        ),
    }
