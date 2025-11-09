"""
Integration and unit tests for the qwen-image-mps CLI.
"""

import os
import sys
import types
from unittest.mock import patch

import pytest

# Ensure `src` is on sys.path before importing the package and stub heavy deps
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Provide a lightweight stub for torch so importing CLI modules doesn't require it
if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    # Minimal attributes used in annotations or device checks
    torch_stub.Generator = type("Generator", (), {})
    torch_stub.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: False)
    )
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_stub.bfloat16 = object()
    torch_stub.float32 = object()
    torch_stub.dtype = object()

    class _DummyCtx:
        def __call__(self, *args, **kwds):
            return self
            
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    torch_stub.inference_mode = lambda: _DummyCtx()
    sys.modules["torch"] = torch_stub

# Provide a minimal stub for huggingface_hub so patching works even if it's not installed
if "huggingface_hub" not in sys.modules:
    hf_stub = types.ModuleType("huggingface_hub")
    # Define attributes that tests will patch or access
    def _dummy(*args, **kwargs):
        raise RuntimeError("huggingface_hub is not available in test stub")

    hf_stub.hf_hub_download = _dummy
    hf_stub.list_repo_files = lambda *args, **kwargs: []
    sys.modules["huggingface_hub"] = hf_stub

from qwen_image_mps.cli import main, get_lora_path
from qwen_image_mps.prompts import sanitize_prompt_for_filename


# Test sanitize_prompt_for_filename function

@pytest.mark.parametrize(
    "prompt, max_length, expected",
    [
        ("A beautiful landscape", 50, "A beautiful landscape"),
        (
            'A "beautiful" landscape with <mountains> and lakes / rivers \ forests',
            50,
            "A beautiful landscape with mountains and lakes riv",
        ),
        (
            "A very long prompt that should be truncated because it exceeds the maximum length",
            20,
            "A very long prompt t",
        ),
        ("A    prompt   with    multiple    spaces", 50, "A prompt with multiple spaces"),
    ],
)
def test_sanitize_prompt_for_filename(prompt, max_length, expected):
    """Test that sanitize_prompt_for_filename correctly sanitizes and truncates prompts."""
    result = sanitize_prompt_for_filename(prompt, max_length=max_length)
    assert result == expected


# Test get_lora_path function with mocking
@patch("huggingface_hub.hf_hub_download")
def test_get_lora_path_custom_filename(mock_hf_hub_download):
    """Test get_lora_path with a custom filename."""
    custom_filename = "custom-lora.safetensors"
    mock_hf_hub_download.return_value = f"/fake/path/{custom_filename}"

    result = get_lora_path(lightning_lora_filename=custom_filename)

    mock_hf_hub_download.assert_called_with(
        repo_id="lightx2v/Qwen-Image-Lightning",
        filename=custom_filename,
        subfolder=None,
        repo_type="model",
    )
    assert result == f"/fake/path/{custom_filename}"


@patch("huggingface_hub.hf_hub_download")
def test_get_lora_path_ultra_fast(mock_hf_hub_download):
    """Test get_lora_path for ultra-fast mode."""
    expected_filename = "Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors"
    mock_hf_hub_download.return_value = f"/fake/path/{expected_filename}"

    result = get_lora_path(ultra_fast=True)

    mock_hf_hub_download.assert_called_with(
        repo_id="lightx2v/Qwen-Image-Lightning",
        filename=expected_filename,
        subfolder=None,
        repo_type="model",
    )
    assert result == f"/fake/path/{expected_filename}"


@patch("huggingface_hub.hf_hub_download")
def test_get_lora_path_default_fast(mock_hf_hub_download):
    """Test get_lora_path for default fast mode."""
    expected_filename = "Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors"
    mock_hf_hub_download.return_value = f"/fake/path/{expected_filename}"

    result = get_lora_path(ultra_fast=False)

    mock_hf_hub_download.assert_called_with(
        repo_id="lightx2v/Qwen-Image-Lightning",
        filename=expected_filename,
        subfolder=None,
        repo_type="model",
    )
    assert result == f"/fake/path/{expected_filename}"


@patch("huggingface_hub.hf_hub_download")
def test_get_lora_path_edit_mode(mock_hf_hub_download):
    """Test get_lora_path for edit mode."""
    expected_filename = "Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors"
    mock_hf_hub_download.return_value = f"/fake/path/{expected_filename}"

    result = get_lora_path(edit_mode=True)

    mock_hf_hub_download.assert_called_with(
        repo_id="lightx2v/Qwen-Image-Lightning",
        filename=expected_filename,
        subfolder="Qwen-Image-Edit-2509",
        repo_type="model",
    )
    assert result == f"/fake/path/{expected_filename}"


@patch("huggingface_hub.hf_hub_download")
def test_get_lora_path_custom_filename_precedence(mock_hf_hub_download):
    """Test that custom_filename takes precedence over mode flags."""
    custom_filename = "custom-filename.safetensors"
    mock_hf_hub_download.return_value = f"/fake/path/{custom_filename}"

    result = get_lora_path(ultra_fast=True, lightning_lora_filename=custom_filename)

    mock_hf_hub_download.assert_called_with(
        repo_id="lightx2v/Qwen-Image-Lightning",
        filename=custom_filename,
        subfolder=None,
        repo_type="model",
    )
    assert result == f"/fake/path/{custom_filename}"


# Integration tests for the CLI command parsing
@patch("qwen_image_mps._cli.main.generate_image")
def test_cli_generate_defaults(mock_generate_image):
    """Test that the CLI calls generate_image with default arguments."""
    # Mock sys.argv
    test_args = ["qwen-image-mps", "generate"]
    with patch.object(sys, "argv", test_args):
        main()

    mock_generate_image.assert_called_once()
    args, _ = mock_generate_image.call_args
    assert args[0].prompt is not None
    assert args[0].fast is False
    assert args[0].ultra_fast is False


@patch("qwen_image_mps._cli.main.generate_image")
def test_cli_generate_custom_args(mock_generate_image):
    """Test that the CLI correctly parses custom arguments for the generate command."""
    test_args = [
        "qwen-image-mps",
        "generate",
        "--prompt",
        "a test prompt",
        "--steps",
        "10",
        "--ultra-fast",
        "--output-filename",
        "my-test-image",
        "--outdir",
        "/tmp/images",
    ]
    with patch.object(sys, "argv", test_args):
        main()

    mock_generate_image.assert_called_once()
    args, _ = mock_generate_image.call_args
    assert args[0].prompt == "a test prompt"
    assert args[0].steps == 10
    assert args[0].ultra_fast is True
    assert args[0].output_filename == "my-test-image"
    assert args[0].output_dir == "/tmp/images"


@patch("qwen_image_mps._cli.main.edit_image")
def test_cli_edit_command(mock_edit_image):
    """Test that the CLI correctly parses arguments for the edit command."""
    test_args = [
        "qwen-image-mps",
        "edit",
        "--input",
        "input.png",
        "--prompt",
        "edit prompt",
        "--fast",
    ]
    with patch.object(sys, "argv", test_args):
        main()

    mock_edit_image.assert_called_once()
    args, _ = mock_edit_image.call_args
    assert args[0].input == ["input.png"]
    assert args[0].prompt == "edit prompt"
    assert args[0].fast is True


@patch("qwen_image_mps._cli.main.generate_image")
def test_cli_backward_compatibility_no_command(mock_generate_image):
    """Test that running without a subcommand defaults to 'generate' for backward compatibility."""
    # This test simulates invoking the script like `qwen-image-mps --prompt "a prompt"`
    test_args = ["qwen-image-mps", "--prompt", "a backward-compatible prompt"]
    with patch.object(sys, "argv", test_args):
        main()

    mock_generate_image.assert_called_once()
    args, _ = mock_generate_image.call_args
    assert args[0].prompt == "a backward-compatible prompt"
