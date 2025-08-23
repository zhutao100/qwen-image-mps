"""
Integration tests for the refactored generate_image function.
Tests the actual function without mocking to ensure everything works end-to-end.
"""

import os
import sys
from argparse import Namespace
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from qwen_image_mps.cli import generate_image, GenerationStep


@pytest.mark.slow
class TestGenerateImageIntegration:
    """Integration tests for the refactored generate_image function."""
    
    def test_cli_backward_compatibility_simple_generation(self):
        """Test: CLI can still generate images with basic arguments (small test)."""
        args = Namespace(
            prompt="Sunset in New York City",
            steps=1,  # Minimal steps for speed
            seed=123,
            num_images=1,
            lora=None,
            batman=True,
            ultra_fast=True,
            fast=False,
            output_path="image-test-compatibility.png"
        )
        
        # Save to project root instead of temp directory
        original_dir = os.getcwd()
        try:
            # Act: Consume the generator
            results = list(generate_image(args))
            
            # Assert: Check that function completes and creates files
            assert len(results) > 0, "Generator should yield results"
            
            # Check for generated image files
            png_files = list(Path('.').glob('*.png'))
            assert len(png_files) >= 1, "Should generate at least one PNG file"
            
            # Verify file is not empty
            for png_file in png_files:
                assert png_file.stat().st_size > 0, f"Generated file {png_file} should not be empty"
                
        finally:
            # Restore original directory
            os.chdir(original_dir)
    
    def test_generator_yields_expected_steps(self):
        """Test: Generator yields all expected GenerationStep events."""
        args = Namespace(
            prompt="Sunset in New York City",
            steps=1,  # Minimal steps for speed
            seed=123,
            num_images=1,
            lora=None,
            batman=True,
            ultra_fast=True,
            fast=False,
            output_path="image-test-output.png"
        )
        
        # Save to project root instead of temp directory
        original_dir = os.getcwd()
        try:
            # Act: Collect yielded steps
            yielded_steps = []
            final_result = None
            
            for result in generate_image(args):
                if isinstance(result, GenerationStep):
                    yielded_steps.append(result)
                    print(f"DEBUG: Yielded step: {result}")
                else:
                    final_result = result
                    print(f"DEBUG: Final result type: {type(result)} - {result}")
            
            # Assert: Verify expected steps are yielded
            # Note: The function may yield additional steps based on internal logic
            required_steps = [
                GenerationStep.INIT,
                GenerationStep.LOADING_MODEL,
                GenerationStep.MODEL_LOADED,
                GenerationStep.PREPARING_GENERATION,
                GenerationStep.INFERENCE_START,
                GenerationStep.INFERENCE_COMPLETE,
                GenerationStep.SAVING_IMAGE,
                GenerationStep.IMAGE_SAVED,
                GenerationStep.COMPLETE
            ]
            
            # Verify required steps are all present
            for step in required_steps:
                assert step in yielded_steps, f"Required step {step} not found in yielded steps"
            
            # Verify the steps are in the right general order (INIT first, COMPLETE last)
            assert yielded_steps[0] == GenerationStep.INIT, "First step should be INIT"
            assert yielded_steps[-1] == GenerationStep.COMPLETE, "Last step should be COMPLETE"
            
            # Verify we have a reasonable number of steps (at least the required ones)
            assert len(yielded_steps) >= len(required_steps), "Should have at least the required number of steps"
            assert final_result is not None, "Should yield final result"
            assert isinstance(final_result, list), "Final result should be a list of paths"
            
        finally:
            # Restore original directory (though not needed for this test)
            os.chdir(original_dir)
    
    
    def test_multiple_images_generation(self):
        """Test: Can generate multiple images."""
        args = Namespace(
            prompt="Sunset in New York City",
            steps=1,  # Minimal steps for speed
            seed=123,
            num_images=2,  # Generate 2 images
            lora=None,
            batman=True,
            ultra_fast=True,
            fast=False
            # Note: No output_path for multiple images - will use auto-generated names
        )
        
        # Save to project root instead of temp directory
        original_dir = os.getcwd()
        try:
            # Act: Get final result
            final_result = None
            for result in generate_image(args):
                if not isinstance(result, GenerationStep):
                    final_result = result
            
            # Assert: Multiple files created
            assert final_result is not None, "Should yield final result"
            assert isinstance(final_result, list), "Final result should be a list"
            assert len(final_result) == 2, "Should have paths for 2 generated images"
            
            # Check actual files exist
            png_files = list(Path('.').glob('*.png'))
            assert len(png_files) >= 2, "Should create at least 2 PNG files"
            
            # Verify paths are absolute and files exist
            for path in final_result:
                assert os.path.isabs(path), f"Path {path} should be absolute"
                assert os.path.exists(path), f"File {path} should exist"
                
        finally:
            # Restore original directory
            os.chdir(original_dir)
    
    