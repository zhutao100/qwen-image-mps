"""
Integration tests for the refactored edit_image function.
Tests the actual function without mocking to ensure everything works end-to-end.
"""

import os
import sys
from argparse import Namespace
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qwen_image_mps.cli import edit_image


@pytest.mark.slow
class TestEditImageIntegration:
    """Integration tests for the refactored edit_image function."""

    def test_edit_image_basic(self):
        """Test: CLI can edit an image with basic arguments."""
        # Ensure the input image exists
        input_image_path = "example.png"
        if not os.path.exists(input_image_path):
            pytest.skip(f"Input image not found at {input_image_path}")

        args = Namespace(
            input=[input_image_path],
            prompt="add a cat to the image",
            steps=1,  # Minimal steps for speed
            seed=123,
            lora=None,
            batman=False,
            ultra_fast=True,
            fast=False,
            output="edited-image-test.png",
            cfg_scale=1.0,
            negative_prompt=None,
            output_dir="output",
            output_filename=None,
            memory_efficient=False,
            quantization=None,
            lightning_lora_filename=None,
        )

        # Run the edit_image function
        edit_image(args)

        # Assert: Check that the output image was created
        output_path = Path("output/edited-image-test.png")
        assert output_path.exists(), "Output image should be created"
        assert output_path.stat().st_size > 0, "Output image should not be empty"
