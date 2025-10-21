"""
SMPL-X mesh generation from parameters.
"""

import os
import torch
import numpy as np
import trimesh
from typing import Dict, List, Optional
from pathlib import Path

import smplx


class SMPLMeshGenerator:
    """Generate meshes from SMPL parameters."""

    def __init__(
        self,
        model_path: str = "models",
        gender: str = "neutral",
        num_betas: int = 10,
        device: str = "cpu",
    ):
        """
        Initialize SMPL mesh generator.

        Args:
            model_path: Path to SMPL model files
            gender: Gender of the model ('neutral', 'male', or 'female')
            num_betas: Number of shape parameters
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.model_path = model_path
        self.gender = gender
        self.num_betas = num_betas
        self.device = torch.device(device)
        self.model = self._load_model()

    def _load_model(self):
        """Load the SMPL model."""
        model = smplx.create(
            self.model_path,
            model_type="smpl",
            gender=self.gender,
            num_betas=self.num_betas,
        ).to(self.device)
        return model

    def generate_mesh(self, smpl_params: Dict) -> trimesh.Trimesh:
        """
        Generate mesh from SMPL parameters.

        Args:
            smpl_params: Dictionary containing SMPL parameters
                - betas: Shape parameters (num_betas,) or (1, num_betas)
                - body_pose: Body pose parameters (69,) or (23, 3) or (1, 69)
                - global_orient: Global orientation (3,) or (1, 3)
                - transl: Translation (3,) or (1, 3) [optional]

        Returns:
            Trimesh object
        """
        # Debug: Print parameter info
        self._debug_params(smpl_params)

        # Convert parameters to tensors and ensure correct shape
        betas = torch.tensor(
            smpl_params.get("betas", np.zeros(self.num_betas)),
            dtype=torch.float32,
            device=self.device,
        )
        if betas.ndim == 1:
            betas = betas.unsqueeze(0)

        body_pose = torch.tensor(
            smpl_params.get("body_pose", np.zeros(69)),
            dtype=torch.float32,
            device=self.device,
        )
        if body_pose.ndim == 1:
            body_pose = body_pose.unsqueeze(0)

        global_orient = torch.tensor(
            smpl_params.get("global_orient", np.zeros(3)),
            dtype=torch.float32,
            device=self.device,
        )
        if global_orient.ndim == 1:
            global_orient = global_orient.unsqueeze(0)

        transl = smpl_params.get("transl", None)
        if transl is not None:
            transl = torch.tensor(transl, dtype=torch.float32, device=self.device)
            if transl.ndim == 1:
                transl = transl.unsqueeze(0)

        # Generate mesh using SMPL model
        with torch.no_grad():
            output = self.model(
                betas=betas,
                body_pose=body_pose,
                global_orient=global_orient,
                transl=transl,
            )

        vertices = output.vertices[0].cpu().numpy()
        faces = self.model.faces

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        return mesh

    def _debug_params(self, smpl_params: Dict):
        """Debug helper to check SMPL parameters."""
        print("\n" + "=" * 50)
        print("SMPL Parameters Debug Info:")
        print("=" * 50)

        # Check what keys are present
        print(f"Available keys: {list(smpl_params.keys())}")

        # Check each parameter
        for key in ["betas", "body_pose", "global_orient", "transl"]:
            if key in smpl_params and smpl_params[key] is not None:
                param = smpl_params[key]
                if isinstance(param, (np.ndarray, list)):
                    param_array = np.array(param)
                    print(f"\n{key}:")
                    print(f"  Shape: {param_array.shape}")
                    print(
                        f"  Min: {param_array.min():.4f}, Max: {param_array.max():.4f}"
                    )
                    print(
                        f"  Mean: {param_array.mean():.4f}, Std: {param_array.std():.4f}"
                    )
                    print(f"  All zeros: {np.allclose(param_array, 0)}")
                    if param_array.size <= 10:
                        print(f"  Values: {param_array}")
                    else:
                        print(f"  First 10 values: {param_array.flatten()[:10]}")
            else:
                print(f"\n{key}: NOT PROVIDED (will use zeros)")

        print("=" * 50 + "\n")

    def check_params(self, smpl_params: Dict) -> Dict[str, bool]:
        """
        Check if SMPL parameters are valid and non-zero.

        Args:
            smpl_params: Dictionary containing SMPL parameters

        Returns:
            Dictionary with validation results
        """
        results = {
            "has_body_pose": False,
            "body_pose_non_zero": False,
            "has_betas": False,
            "has_global_orient": False,
            "has_transl": False,
        }

        # Check body_pose
        if "body_pose" in smpl_params and smpl_params["body_pose"] is not None:
            results["has_body_pose"] = True
            body_pose_array = np.array(smpl_params["body_pose"])
            results["body_pose_non_zero"] = not np.allclose(body_pose_array, 0)

        # Check other params
        if "betas" in smpl_params and smpl_params["betas"] is not None:
            results["has_betas"] = True

        if "global_orient" in smpl_params and smpl_params["global_orient"] is not None:
            results["has_global_orient"] = True

        if "transl" in smpl_params and smpl_params["transl"] is not None:
            results["has_transl"] = True

        return results

    def save_mesh(self, mesh: trimesh.Trimesh, output_path: str):
        """
        Save mesh to file.

        Args:
            mesh: Trimesh object
            output_path: Path to save mesh (e.g., .obj, .ply)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(output_path))

    def generate_meshes_batch(
        self, params_list: List[Dict], output_dir: str
    ) -> List[str]:
        """
        Generate and save meshes for a batch of SMPL parameters.

        Args:
            params_list: List of SMPL parameter dictionaries
            output_dir: Directory to save meshes

        Returns:
            List of paths to saved meshes
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        mesh_paths = []

        for i, params in enumerate(params_list):
            mesh = self.generate_mesh(params)
            mesh_path = output_dir / f"mesh_{i:06d}.obj"
            self.save_mesh(mesh, str(mesh_path))
            mesh_paths.append(str(mesh_path))

        return mesh_paths


class SMPLXMeshGenerator:
    """Generate meshes from SMPL-X parameters."""

    def __init__(
        self,
        model_path: str,
        gender: str = "neutral",
        num_betas: int = 10,
        num_expression_coeffs: int = 10,
    ):
        """
        Initialize SMPL-X mesh generator.

        Args:
            model_path: Path to SMPL-X model files
            gender: Gender of the model ('neutral', 'male', or 'female')
            num_betas: Number of shape parameters
            num_expression_coeffs: Number of expression coefficients
        """
        self.model_path = model_path
        self.gender = gender
        self.num_betas = num_betas
        self.num_expression_coeffs = num_expression_coeffs
        self.model = None

    def load_model(self):
        """Load the SMPL-X model."""
        # Note: This is a placeholder for actual SMPL-X model loading
        # In practice, you would load the actual SMPL-X model here
        # import smplx
        # self.model = smplx.create(
        #     self.model_path,
        #     model_type='smplx',
        #     gender=self.gender,
        #     num_betas=self.num_betas,
        #     num_expression_coeffs=self.num_expression_coeffs,
        #     use_face_contour=False
        # )

        print(f"SMPL-X model would be loaded from {self.model_path}")
        print(
            f"Gender: {self.gender}, Betas: {self.num_betas}, Expression coeffs: {self.num_expression_coeffs}"
        )

    def generate_mesh(self, smplx_params: Dict) -> trimesh.Trimesh:
        """
        Generate mesh from SMPL-X parameters.

        Args:
            smplx_params: Dictionary containing SMPL-X parameters

        Returns:
            Trimesh object
        """
        if self.model is None:
            self.load_model()

        # Convert parameters to tensors
        # Note: This is a placeholder for actual mesh generation
        # In practice, you would use the actual SMPL-X model here
        # output = self.model(
        #     betas=torch.tensor(smplx_params['betas']).unsqueeze(0),
        #     body_pose=torch.tensor(smplx_params['body_pose']).unsqueeze(0),
        #     global_orient=torch.tensor(smplx_params['global_orient']).unsqueeze(0),
        #     transl=torch.tensor(smplx_params['transl']).unsqueeze(0),
        #     left_hand_pose=torch.tensor(smplx_params['left_hand_pose']).unsqueeze(0),
        #     right_hand_pose=torch.tensor(smplx_params['right_hand_pose']).unsqueeze(0),
        #     jaw_pose=torch.tensor(smplx_params['jaw_pose']).unsqueeze(0),
        #     leye_pose=torch.tensor(smplx_params['leye_pose']).unsqueeze(0),
        #     reye_pose=torch.tensor(smplx_params['reye_pose']).unsqueeze(0),
        #     expression=torch.tensor(smplx_params['expression']).unsqueeze(0),
        # )

        # For now, create a dummy mesh (simple cube)
        vertices = self._get_dummy_vertices()
        faces = self._get_dummy_faces()

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        return mesh

    def _get_dummy_vertices(self) -> np.ndarray:
        """Generate dummy vertices for testing."""
        # Simple cube vertices
        vertices = np.array(
            [
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
                [0.5, 0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, 0.5],
                [-0.5, 0.5, 0.5],
            ]
        )
        return vertices

    def _get_dummy_faces(self) -> np.ndarray:
        """Generate dummy faces for testing."""
        # Cube faces
        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],  # Front
                [4, 5, 6],
                [4, 6, 7],  # Back
                [0, 1, 5],
                [0, 5, 4],  # Bottom
                [2, 3, 7],
                [2, 7, 6],  # Top
                [0, 3, 7],
                [0, 7, 4],  # Left
                [1, 2, 6],
                [1, 6, 5],  # Right
            ]
        )
        return faces

    def save_mesh(self, mesh: trimesh.Trimesh, output_path: str):
        """
        Save mesh to file.

        Args:
            mesh: Trimesh object
            output_path: Path to save mesh (e.g., .obj, .ply)
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        mesh.export(output_path)
        print(f"Saved mesh to {output_path}")

    def generate_meshes_batch(
        self, params_list: List[Dict], output_dir: str
    ) -> List[str]:
        """
        Generate and save meshes for a batch of SMPL-X parameters.

        Args:
            params_list: List of SMPL-X parameter dictionaries
            output_dir: Directory to save meshes

        Returns:
            List of paths to saved meshes
        """
        os.makedirs(output_dir, exist_ok=True)
        mesh_paths = []

        for i, params in enumerate(params_list):
            mesh = self.generate_mesh(params)
            mesh_path = os.path.join(output_dir, f"mesh_{i:06d}.obj")
            self.save_mesh(mesh, mesh_path)
            mesh_paths.append(mesh_path)

        return mesh_paths
