"""
Pytest configuration and fixtures for qwen-image-mps tests.
"""

import os
import sys

import pytest

# Add src directory to Python path for test imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
