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

    def compute_gaussian_coord2d(self, feature_map, pred, img_shape: tuple = None):
        """The function maps predicted 2d coordinates to batched per-gaussian 2D centers.

        Assumes at most 1 person per image;
        handles v2d(list) with elements of shape (P,Nv,2).

        Returns:
            Tensor of shape (B, N, 2) with Gaussian centers.
        """
        B = feature_map.shape[0]  # Batch size
        N = int(self._avatar.total_gaussians_num)  # Number of Gaussians
        K = int(
            self._avatar.barycentric_coords.shape[0]
        )  # Number of Gaussian per face, default 4

        parents = self._avatar.parents  # (N,3)
        bary = self._avatar.barycentric_coords  # (K,3)

        # TODO(future): handle multiple people per image
        v2d_list = pred["vertices2d"]  # A list of length B with (P, Nv, 2)
        vertices2d = torch.stack([v2d_raw[0] for v2d_raw in v2d_list], dim=0)
        assert vertices2d.shape[0] == B, "Batch size mismatch in vertices2d"

        device = feature_map.device

        parents = parents.to(device=device, dtype=torch.long)  # (N,3)
        bary = bary.to(device=device, dtype=torch.float32)  # (K,3)
        verts2d = vertices2d.to(device=device)  # (B,Nv,2)

        idx = torch.arange(N, device=device) % K  # (N,)
        bary_per_gauss = bary[idx]  # (N,3)

        flat_idx = parents.reshape(-1)  # (N*3,)
        verts_sel = verts2d.index_select(1, flat_idx)  # (B, N*3, 2)
        face_verts = verts_sel.reshape(B, N, 3, 2)  # (B, N, 3, 2)

        centers2d = torch.einsum("bnvc, nv->bnc", face_verts, bary_per_gauss)

        # centers2d are in pixel coordinates (x,y) relative to the original image
        # Return them unchanged; caller (feature_sample) will normalize using the
        # original image shape so grid_sample receives correct coordinates.
        return centers2d  # (B,N,2)

    def compute_gaussian_coord3d(self, feature_map, pred):
        """Batched per-gaussian 3D centers.

        Assumes at most 1 person per image when a people dimension exists.

        Returns:
            Tensor of shape (B, N, 3).
        """
        B = feature_map.shape[0]
        N = int(self._avatar.total_gaussians_num)
        K = int(self._avatar.barycentric_coords.shape[0])

        parents = self._avatar.parents  # (N,3)
        bary = self._avatar.barycentric_coords  # (K,3)

        v3d_list = pred["vertices3d"]
        vertices3d = torch.stack(
            [v3d_raw[0] for v3d_raw in v3d_list], dim=0
        )  # Assume only 1 person, get (B, Nv, 3)
        assert vertices3d.shape[0] == B, "Batch size mismatch in vertices3d"

        device = feature_map.device

        # normalize types/devices
        parents = parents.to(device=device, dtype=torch.long)
        bary = bary.to(device=device, dtype=torch.float32)
        verts3d = vertices3d.to(device=device)

        # Map each gaussian to its barycentric row
        idx = torch.arange(N, device=device) % K  # (N,)
        bary_per_gauss = bary[idx]  # (N,3)

        # Gather parent vertices using advanced indexing
        flat_idx = parents.reshape(-1)  # (N*3,)
        verts_sel = verts3d.index_select(1, flat_idx)  # (B, N*3, 3)
        face_verts = verts_sel.reshape(B, N, 3, 3)  # (B, N, 3(parents), 3(coords))

        # Weighted sum over parents (p) -> (B, N, 3)
        # einsum form: b n p c, n p -> b n c
        centers3d = torch.einsum("bnpc,np->bnc", face_verts, bary_per_gauss)

        return centers3d  # (B,N,3)

    def feature_sample(self, feature_map, pred, img_shape: tuple = None):
        # TODO: Add occlusion handling using coord3d
        """Sample per-Gaussian local features from a feature map (batched).

        Returns: (B, N, C)
        """
        centers2d = self.compute_gaussian_coord2d(
            feature_map, pred, img_shape=img_shape
        )

        B, C, Hf, Wf = feature_map.shape

        # Ensure centers are on the same device/dtype as the feature map
        centers2d = centers2d.to(device=feature_map.device, dtype=feature_map.dtype)

        # Determine original image size to normalize coordinates. If not provided,
        # fall back to assuming the feature map was produced from an image with
        # same spatial dims (not ideal but backward compatible).
        if img_shape is not None:
            H_img, W_img = int(img_shape[0]), int(img_shape[1])
        else:
            # Approximate by scaling from feature map to image space
            H_img, W_img = Hf, Wf

        # Convert pixel coordinates in image space -> normalized grid in [-1,1]
        # Note: using image size here correctly maps original-image pixels into
        # normalized coordinates compatible with feature_map sampling because of
        # the proportional scaling between image and feature map.
        x = centers2d[..., 0] / (W_img - 1) * 2 - 1  # (B,N)
        y = centers2d[..., 1] / (H_img - 1) * 2 - 1  # (B,N)
        grid = torch.stack([x, y], dim=-1).unsqueeze(1)  # (B,1,N,2)

        # grid_sample requires input and grid to have the same dtype; ensure that
        # here (especially important when feature_map is float16).
        if grid.dtype != feature_map.dtype:
            grid = grid.to(dtype=feature_map.dtype)

        sampled = F.grid_sample(
            feature_map,  # (B,C,H,W)
            grid,  # (B,1,N,2)
            mode="bilinear",
            align_corners=True,
        )  # (B,C,1,N)

        return sampled[:, :, 0, :].permute(0, 2, 1)  # (B,N,C)

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
