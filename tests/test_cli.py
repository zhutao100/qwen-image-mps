"""
Integration and unit tests for the qwen-image-mps CLI.
"""

import pytest
from unittest.mock import patch, MagicMock
import os
import sys
# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from qwen_image_mps.cli import (
    main,
    sanitize_prompt_for_filename,
    get_lora_path,
)

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
        repo_type="model",
    )
    assert result == f"/fake/path/{custom_filename}"


@patch("huggingface_hub.hf_hub_download")
def test_get_lora_path_ultra_fast(mock_hf_hub_download):
    """Test get_lora_path for ultra-fast mode."""
    expected_filename = "Qwen-Image-Lightning-4steps-V1.0-bf16.safetensors"
    mock_hf_hub_download.return_value = f"/fake/path/{expected_filename}"

    result = get_lora_path(ultra_fast=True)

    mock_hf_hub_download.assert_called_with(
        repo_id="lightx2v/Qwen-Image-Lightning",
        filename=expected_filename,
        repo_type="model",
    )
    assert result == f"/fake/path/{expected_filename}"


@patch("huggingface_hub.hf_hub_download")
def test_get_lora_path_default_fast(mock_hf_hub_download):
    """Test get_lora_path for default fast mode."""
    expected_filename = "Qwen-Image-Lightning-8steps-V1.1-bf16.safetensors"
    mock_hf_hub_download.return_value = f"/fake/path/{expected_filename}"

    result = get_lora_path(ultra_fast=False)

    mock_hf_hub_download.assert_called_with(
        repo_id="lightx2v/Qwen-Image-Lightning",
        filename=expected_filename,
        repo_type="model",
    )
    assert result == f"/fake/path/{expected_filename}"


@patch("huggingface_hub.hf_hub_download")
def test_get_lora_path_edit_mode(mock_hf_hub_download):
    """Test get_lora_path for edit mode."""
    expected_filename = "Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors"
    mock_hf_hub_download.return_value = f"/fake/path/{expected_filename}"

    result = get_lora_path(edit_mode=True)

    mock_hf_hub_download.assert_called_with(
        repo_id="lightx2v/Qwen-Image-Lightning",
        filename=expected_filename,
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
        repo_type="model",
    )
    assert result == f"/fake/path/{custom_filename}"


# Integration tests for the CLI command parsing
@patch("qwen_image_mps.cli.generate_image")
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


@patch("qwen_image_mps.cli.generate_image")
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


@patch("qwen_image_mps.cli.edit_image")
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
    assert args[0].input == "input.png"
    assert args[0].prompt == "edit prompt"
    assert args[0].fast is True
