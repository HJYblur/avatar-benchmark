"""
Mesh rendering module for visualizing SMPL-X meshes.
"""
import os
import numpy as np
import trimesh
import pyrender
from typing import Tuple, List, Optional
from pathlib import Path
import cv2


class MeshRenderer:
    """Render meshes to images."""
    
    def __init__(self, image_size: Tuple[int, int] = (512, 512),
                 camera_distance: float = 2.5,
                 light_intensity: float = 3.0,
                 background_color: Tuple[int, int, int] = (0, 0, 0)):
        """
        Initialize mesh renderer.
        
        Args:
            image_size: Size of rendered images (width, height)
            camera_distance: Distance of camera from the mesh
            light_intensity: Intensity of the light source
            background_color: Background color (R, G, B)
        """
        self.image_size = image_size
        self.camera_distance = camera_distance
        self.light_intensity = light_intensity
        self.background_color = np.array(background_color) / 255.0
        
    def render_mesh(self, mesh: trimesh.Trimesh, 
                    rotation: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Render a single mesh to an image.
        
        Args:
            mesh: Trimesh object to render
            rotation: Optional rotation matrix to apply to mesh
            
        Returns:
            Rendered image as numpy array (H, W, 3)
        """
        # Create scene
        scene = pyrender.Scene(bg_color=self.background_color, ambient_light=(0.3, 0.3, 0.3))
        
        # Apply rotation if provided
        if rotation is not None:
            mesh.apply_transform(rotation)
        
        # Add mesh to scene
        mesh_node = pyrender.Mesh.from_trimesh(mesh, smooth=True)
        scene.add(mesh_node)
        
        # Setup camera
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        camera_pose = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, self.camera_distance],
            [0.0, 0.0, 0.0, 1.0],
        ])
        scene.add(camera, pose=camera_pose)
        
        # Setup light
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], 
                                         intensity=self.light_intensity)
        scene.add(light, pose=camera_pose)
        
        # Render
        renderer = pyrender.OffscreenRenderer(self.image_size[0], self.image_size[1])
        color, _ = renderer.render(scene)
        renderer.delete()
        
        return color
    
    def render_meshes_batch(self, mesh_paths: List[str], 
                           output_dir: str) -> List[str]:
        """
        Render a batch of meshes to images.
        
        Args:
            mesh_paths: List of paths to mesh files
            output_dir: Directory to save rendered images
            
        Returns:
            List of paths to rendered images
        """
        os.makedirs(output_dir, exist_ok=True)
        render_paths = []
        
        for i, mesh_path in enumerate(mesh_paths):
            # Load mesh
            mesh = trimesh.load(mesh_path)
            
            # Render mesh
            rendered = self.render_mesh(mesh)
            
            # Save rendered image
            render_path = os.path.join(output_dir, f"render_{i:06d}.png")
            cv2.imwrite(render_path, cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
            render_paths.append(render_path)
            
            if (i + 1) % 10 == 0:
                print(f"Rendered {i + 1}/{len(mesh_paths)} meshes")
        
        print(f"Rendered {len(mesh_paths)} meshes to {output_dir}")
        return render_paths
    
    def render_with_overlay(self, mesh: trimesh.Trimesh, 
                           background_image: np.ndarray) -> np.ndarray:
        """
        Render mesh with background image overlay.
        
        Args:
            mesh: Trimesh object to render
            background_image: Background image to overlay on
            
        Returns:
            Rendered image with overlay
        """
        # Render mesh
        rendered = self.render_mesh(mesh)
        
        # Resize background if needed
        if background_image.shape[:2] != rendered.shape[:2]:
            background_image = cv2.resize(background_image, 
                                         (rendered.shape[1], rendered.shape[0]))
        
        # Simple alpha blending (assuming black background for mesh)
        mask = np.all(rendered == 0, axis=2, keepdims=True)
        overlay = np.where(mask, background_image, rendered)
        
        return overlay
