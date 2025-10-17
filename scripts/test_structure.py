#!/usr/bin/env python3
"""
Test script to verify project structure and imports.
"""
import os
import sys
from pathlib import Path

def check_directory_structure():
    """Check if all required directories exist."""
    print("Checking directory structure...")
    
    base_dir = Path(__file__).parent.parent
    required_dirs = [
        "configs",
        "data/videos",
        "data/processed",
        "models/smplx",
        "models/pymafx",
        "outputs/meshes",
        "outputs/renders",
        "outputs/metrics",
        "src/preprocessing",
        "src/inference",
        "src/rendering",
        "src/metrics",
        "src/utils",
        "scripts",
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        full_path = base_dir / dir_path
        exists = full_path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {dir_path}")
        if not exists:
            all_exist = False
    
    return all_exist


def check_files():
    """Check if all required files exist."""
    print("\nChecking required files...")
    
    base_dir = Path(__file__).parent.parent
    required_files = [
        "requirements.txt",
        "setup.py",
        "configs/default.yaml",
        "scripts/run_pipeline.py",
        "scripts/example.py",
        "src/__init__.py",
        "src/preprocessing/__init__.py",
        "src/preprocessing/video_preprocessor.py",
        "src/inference/__init__.py",
        "src/inference/pymafx_wrapper.py",
        "src/rendering/__init__.py",
        "src/rendering/mesh_generator.py",
        "src/rendering/renderer.py",
        "src/metrics/__init__.py",
        "src/metrics/evaluator.py",
        "src/utils/__init__.py",
        "src/utils/config_utils.py",
    ]
    
    all_exist = True
    for file_path in required_files:
        full_path = base_dir / file_path
        exists = full_path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {file_path}")
        if not exists:
            all_exist = False
    
    return all_exist


def check_module_structure():
    """Check if Python modules can be imported."""
    print("\nChecking module structure (without dependencies)...")
    
    base_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(base_dir))
    
    modules = [
        "src",
        "src.preprocessing",
        "src.inference",
        "src.rendering",
        "src.metrics",
        "src.utils",
    ]
    
    all_ok = True
    for module_name in modules:
        try:
            __import__(module_name)
            print(f"  ✓ {module_name}")
        except Exception as e:
            print(f"  ✗ {module_name}: {e}")
            all_ok = False
    
    return all_ok


def main():
    """Run all checks."""
    print("=" * 60)
    print("Avatar Benchmark - Structure Validation")
    print("=" * 60)
    print()
    
    dir_ok = check_directory_structure()
    files_ok = check_files()
    modules_ok = check_module_structure()
    
    print("\n" + "=" * 60)
    if dir_ok and files_ok and modules_ok:
        print("✓ All checks passed!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Download SMPL-X models to models/smplx/")
        print("3. Download PyMAF-X checkpoint to models/pymafx/")
        print("4. Place input videos in data/videos/")
        print("5. Run: python scripts/run_pipeline.py --video data/videos/input.mp4")
    else:
        print("✗ Some checks failed!")
        return 1
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
