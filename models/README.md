# Models Directory

This directory contains the SMPL-X and PyMAF-X model files.

## Structure

- `smplx/`: SMPL-X model files
- `pymafx/`: PyMAF-X model checkpoint

## Setup

1. **SMPL-X Models**: 
   - Download SMPL-X models from [https://smpl-x.is.tue.mpg.de/](https://smpl-x.is.tue.mpg.de/)
   - Place model files (SMPLX_NEUTRAL.npz, SMPLX_MALE.npz, SMPLX_FEMALE.npz) in `models/smplx/`

2. **PyMAF-X Model**:
   - Download PyMAF-X checkpoint from [https://github.com/HongwenZhang/PyMAF-X](https://github.com/HongwenZhang/PyMAF-X)
   - Place checkpoint file in `models/pymafx/`

## Required Files

```
models/
├── smplx/
│   ├── SMPLX_NEUTRAL.npz
│   ├── SMPLX_MALE.npz
│   └── SMPLX_FEMALE.npz
└── pymafx/
    └── checkpoint.pth
```
