"""Custom build script for gooviz."""

import os
import shutil
from pathlib import Path

def custom_build(app, source_dir, build_dir, target_dir, **kwargs):
    """Custom build steps for the package."""
    # Create necessary directories
    os.makedirs(target_dir, exist_ok=True)
    
    # Copy additional files
    files_to_copy = [
        "LICENSE",
        "README.md",
    ]
    
    for file in files_to_copy:
        if os.path.exists(file):
            shutil.copy2(file, os.path.join(target_dir, file))
    
    # Create __init__.py in examples directory if it doesn't exist
    examples_init = Path("examples/__init__.py")
    if not examples_init.exists():
        examples_init.parent.mkdir(exist_ok=True)
        examples_init.write_text('"""Example scripts for goo-visualization."""\n') 