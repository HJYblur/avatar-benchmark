import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder.avatar_template import AvatarTemplate


class AvatarGaussianEstimator(nn.Module):
    """
    Feature Sampler features and predicts Gaussian parameters
    """

    def __init__(self, template: AvatarTemplate):
        super().__init__()
        self._avatar = template

    @property
    def template(self) -> AvatarTemplate:
        return self._avatar

    def compute_gaussian_coord2d(self, feature_map, pred):
        """Vectorized computation of per-gaussian 2D centers.

        Returns:
            Tensor of shape (N,2) with Gaussian centers in the same coord space as `self._vertices2d`.
        """
        # Total count of gaussians in the template and per-face gaussian count
        N = int(self._avatar.total_gaussians_num)
        K = int(self._avatar.barycentric_coords.shape[0])

        # Retrieve parents and barycentric coords from the template.
        parents = self._avatar.parents
        bary = self._avatar.barycentric_coords

        # Load predicted coords for vertices, support shapes:
        #   (B, P, Nv, 2), (B, Nv, 2), or (Nv, 2)
        v2d = pred["vertices2d"]
        if v2d.dim() == 4:
            vertices2d = v2d[0, 0]  # assume first person
        elif v2d.dim() == 3:
            vertices2d = v2d[0]
        else:
            vertices2d = v2d
        # vertices3d = pred["vertices3d"][0]  # (Nv, 3)  # kept for future use
        device = feature_map.device

        # Normalize types and devices
        parents = parents.to(device=device, dtype=torch.long)
        bary = bary.to(device=device, dtype=torch.float32)
        verts2d = vertices2d.to(device=device)

        # Create an index that maps each gaussian (0..N-1) to its barycentric row (0..K-1)
        idx = torch.arange(N, device=device) % K  # (N,)
        bary_per_gauss = bary[idx]  # (N,3)

        # parents: (N,3) -> gather vertex coordinates: face_verts (N,3,2)
        face_verts = verts2d[parents]  # (N,3,2)

        # Weighted sum over the 3 parent vertices using barycentric weights -> (N,2)
        centers2d = (face_verts * bary_per_gauss.unsqueeze(-1)).sum(dim=1)

        return centers2d

    def compute_gaussian_coord3d(self, feature_map, pred):
        """Vectorized computation of per-gaussian 3D centers.

        Returns:
            Tensor of shape (N,3) with Gaussian centers in the same coord space as `self._vertices3d`.
        """
        N = int(self._avatar.total_gaussians_num)
        K = int(self._avatar.barycentric_coords.shape[0])

        # Retrieve parents and barycentric coords from the template.
        parents = self._avatar.parents
        bary = self._avatar.barycentric_coords

        # Load predicted coords for vertices (support (B,P,Nv,3), (B,Nv,3), (Nv,3))
        v3d = pred["vertices3d"]
        if v3d.dim() == 4:
            vertices3d = v3d[0, 0]
        elif v3d.dim() == 3:
            vertices3d = v3d[0]
        else:
            vertices3d = v3d
        device = feature_map.device

        # Normalize types and devices
        parents = parents.to(device=device, dtype=torch.long)
        bary = bary.to(device=device, dtype=torch.float32)
        verts3d = vertices3d.to(device=device)

        # Create an index that maps each gaussian (0..N-1) to its barycentric row (0..K-1)
        idx = torch.arange(N, device=device) % K  # (N,)
        bary_per_gauss = bary[idx]  # (N,3)

        # parents: (N,3) -> gather vertex coordinates: face_verts (N,3,3)
        face_verts = verts3d[parents]  # (N,3,3)

        # Weighted sum over the 3 parent vertices using barycentric weights -> (N,3)
        centers3d = (face_verts * bary_per_gauss.unsqueeze(-1)).sum(dim=1)

        return centers3d

    def feature_sample(self, feature_map, pred):
        centers2d = self.compute_gaussian_coord2d(feature_map, pred)  # (N,2)
        B, C, H, W = feature_map.shape
        device = feature_map.device

        # Normalize
        x = centers2d[:, 0] / (W - 1) * 2 - 1
        y = centers2d[:, 1] / (H - 1) * 2 - 1
        centers_norm = torch.stack([x, y], dim=-1)  # (N,2)

        # Build grid: (B,1,N,2)
        grid = centers_norm.unsqueeze(0).unsqueeze(1).expand(B, 1, -1, 2)

        # Sample all N points
        # grid sample expects feature map (B, C, H, W) and grid of shape (B, H_out, W_out, 2)
        sampled = F.grid_sample(
            feature_map,  # (B,C,H,W)
            grid,  # (B,1,N,2)
            mode="bilinear",
            align_corners=True,
        )

        # Now sampled = (B,C,1,N)
        # Return (B,N,C)
        return sampled[:, :, 0, :].permute(0, 2, 1)

    # def forward(self, feature_map, pred, gaussian_estimator):
    #     """Predict Gaussian parameters from the sampled features.

    #     Returns:
    #         A tensor of shape (N, 1 + 4 + 3 + 3) containing Gaussian parameters:
    #           - 'scale': Tensor of shape (N, 3)
    #           - 'rotation': Tensor of shape (N, 4)
    #           - 'alpha': Tensor of shape (N,)
    #           - 'sh': Tensor of shape (N, 3)
    #     """
    #     batch_features = self.feature_sample(feature_map, pred)  # (B,N,C)
    #     # For now, assume batch size 1
    #     batch_gaussian_params = gaussian_estimator(batch_features[0])  # (N, P)
    #     return batch_gaussian_params
