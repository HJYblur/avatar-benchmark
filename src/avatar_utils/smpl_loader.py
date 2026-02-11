import smplx
import torch
import numpy as np
from typing import Dict, Any, Optional


def smplx_params_to_vertices(
    smplx_params: Dict[str, Any],
    model_path: str = "./models",
    gender: str = "neutral",
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Convert SMPL-X parameters to vertex coordinates.
    
    Args:
        smplx_params: Dictionary containing SMPL-X parameters with keys like:
            - 'betas': shape parameters (10,) or (1, 10)
            - 'body_pose': body joint rotations (63,) or (1, 63) or (21, 3)
            - 'global_orient': global rotation (3,) or (1, 3)
            - 'translation' or 'transl': translation (3,) or (1, 3)
            - Optional: 'left_hand_pose', 'right_hand_pose', 'jaw_pose', 
                       'leye_pose', 'reye_pose', 'expression', 'scale'
        model_path: Path to SMPL-X model files directory
        gender: 'neutral', 'male', or 'female'
        device: Device to run computation on
        
    Returns:
        vertices: (V, 3) tensor of vertex coordinates
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Detect if hand poses use PCA (check if left_hand_pose exists and its size)
    use_pca = False
    num_pca_comps = 12
    if 'left_hand_pose' in smplx_params:
        hand_pose = smplx_params['left_hand_pose']
        if isinstance(hand_pose, np.ndarray):
            hand_size = hand_pose.shape[-1] if hand_pose.ndim > 0 else 0
        elif isinstance(hand_pose, torch.Tensor):
            hand_size = hand_pose.shape[-1] if hand_pose.ndim > 0 else 0
        else:
            hand_size = len(hand_pose) if hasattr(hand_pose, '__len__') else 0
        
        if hand_size > 0 and hand_size < 45:
            use_pca = True
            num_pca_comps = hand_size
    
    # Create SMPL-X model with appropriate settings
    smplx_model = smplx.create(
        model_path=model_path,
        model_type="smplx",
        gender=gender,
        num_betas=10,
        use_pca=use_pca,
        num_pca_comps=num_pca_comps if use_pca else 45,
        batch_size=1
    ).to(device)
    
    # Helper function to convert params to correct shape and device
    def to_tensor(key: str, default_shape: tuple) -> torch.Tensor:
        if key in smplx_params:
            val = smplx_params[key]
            # Handle trimesh TrackedArray
            if hasattr(val, '__array__'):
                val = np.asarray(val)
            
            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val)
            elif not isinstance(val, torch.Tensor):
                val = torch.tensor(val)
            
            # Ensure float32
            val = val.float()
            
            # Reshape if needed (flatten then reshape to batch format)
            if val.dim() == 1:
                val = val.reshape(1, -1)
            elif val.dim() > 2:
                # e.g., (21, 3) -> (1, 63)
                val = val.reshape(1, -1)
                
            return val.to(device)
        else:
            return torch.zeros(default_shape, device=device, dtype=torch.float32)
    
    # Extract parameters with appropriate defaults
    betas = to_tensor('betas', (1, 10))
    global_orient = to_tensor('global_orient', (1, 3))
    body_pose = to_tensor('body_pose', (1, 63))  # 21 joints * 3
    
    # Hand poses - size depends on use_pca
    hand_shape = (1, num_pca_comps) if use_pca else (1, 45)
    left_hand_pose = to_tensor('left_hand_pose', hand_shape)
    right_hand_pose = to_tensor('right_hand_pose', hand_shape)
    
    jaw_pose = to_tensor('jaw_pose', (1, 3))
    leye_pose = to_tensor('leye_pose', (1, 3))
    reye_pose = to_tensor('reye_pose', (1, 3))
    expression = to_tensor('expression', (1, 10))
    
    # Try 'translation' first, then 'transl'
    transl = None
    if 'translation' in smplx_params:
        transl = to_tensor('translation', (1, 3))
    else:
        transl = to_tensor('transl', (1, 3))
    
    # Extract scale if present (scalar)
    scale_val = 1.0
    if 'scale' in smplx_params:
        scale_param = smplx_params['scale']
        if isinstance(scale_param, (np.ndarray, torch.Tensor)):
            scale_val = float(scale_param.flatten()[0])
        else:
            scale_val = float(scale_param)
    
    # Forward pass through SMPL-X model
    with torch.no_grad():
        output = smplx_model(
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            expression=expression,
            transl=transl,
            return_verts=True
        )
    
    vertices = output.vertices[0]  # (V, 3) - remove batch dimension
    
    # Apply scale if present
    if scale_val != 1.0:
        vertices = vertices * scale_val
    
    return vertices