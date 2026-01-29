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

        device = feature_map.device

        parents = parents.to(device=device, dtype=torch.long)  # (N,3)
        bary = bary.to(device=device, dtype=torch.float32)  # (K,3)
        verts2d = self._stack_vertices2d(pred, device=device)  # (B,Nv,2)
        assert verts2d.shape[0] == B, "Batch size mismatch in vertices2d"

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

        device = feature_map.device

        # normalize types/devices
        parents = parents.to(device=device, dtype=torch.long)
        bary = bary.to(device=device, dtype=torch.float32)
        verts3d = self._stack_vertices3d(pred, device=device)
        assert verts3d.shape[0] == B, "Batch size mismatch in vertices3d"

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

    def _stack_vertices2d(self, pred, device):
        cached = pred.get("_vertices2d_stack")
        if cached is None:
            v2d_list = pred["vertices2d"]  # A list of length B with (P, Nv, 2)
            cached = torch.stack([v2d_raw[0] for v2d_raw in v2d_list], dim=0)
            pred["_vertices2d_stack"] = cached
        return cached.to(device=device)

    def _stack_vertices3d(self, pred, device):
        cached = pred.get("_vertices3d_stack")
        if cached is None:
            v3d_list = pred["vertices3d"]  # A list of length B with (P, Nv, 3)
            cached = torch.stack([v3d_raw[0] for v3d_raw in v3d_list], dim=0)
            pred["_vertices3d_stack"] = cached  # (B, Nv, 3)
        return cached.to(device=device)

    def compute_gaussian_normals(self, pred, device):
        """Compute per-Gaussian normals from mesh face vertices."""
        vertices3d = self._stack_vertices3d(pred, device=device)
        B = vertices3d.shape[0]
        N = int(self._avatar.total_gaussians_num)
        parents = self._avatar.parents.to(device=device, dtype=torch.long)  # (N,3)

        flat_idx = parents.reshape(-1)  # (N*3,)
        verts_sel = vertices3d.index_select(dim=1, index=flat_idx)  # (B, N*3, 3)
        face_verts = verts_sel.reshape(B, N, 3, 3)  # (B, N, 3, 3)

        e1 = face_verts[:, :, 1] - face_verts[:, :, 0]
        e2 = face_verts[:, :, 2] - face_verts[:, :, 0]
        normals = torch.linalg.cross(e1, e2, dim=-1)
        normals = normals / (
            torch.norm(normals, dim=-1, keepdim=True) + 1e-8
        )  # (B,N,3)
        return normals

    def build_depth_map(self, pred, img_shape: tuple, device):
        """Approximate per-view depth map from projected mesh vertices."""
        H_img, W_img = int(img_shape[0]), int(img_shape[1])
        vertices2d = self._stack_vertices2d(pred, device=device)
        vertices3d = self._stack_vertices3d(pred, device=device)

        B, Nv, _ = vertices2d.shape
        depth_maps = torch.full(
            (B, H_img * W_img),
            float("inf"),
            device=device,
            dtype=vertices3d.dtype,
        )

        x = torch.round(vertices2d[..., 0]).to(torch.long)
        y = torch.round(vertices2d[..., 1]).to(torch.long)
        z = vertices3d[..., 2]

        valid = (x >= 0) & (x < W_img) & (y >= 0) & (y < H_img)
        for b in range(B):
            if not valid[b].any():
                continue
            idx = (y[b, valid[b]] * W_img + x[b, valid[b]]).to(torch.long)
            depth_vals = z[b, valid[b]]
            # Keep the minimum depth per pixel
            if hasattr(torch.Tensor, "scatter_reduce_"):
                depth_maps[b].scatter_reduce_(
                    0, idx, depth_vals, reduce="amin", include_self=True
                )
            else:  # pragma: no cover - compatibility fallback
                for i in range(idx.numel()):
                    depth_maps[b, idx[i]] = torch.minimum(
                        depth_maps[b, idx[i]], depth_vals[i]
                    )

        return depth_maps.view(B, H_img, W_img)

    def compute_view_weights(
        self,
        feature_map,
        pred,
        centers2d: torch.Tensor,
        centers3d: torch.Tensor,
        img_shape: tuple,
        depth_eps: float = 1e-3,
    ):
        """Compute per-view weights using view angle and occlusion depth test."""
        device = feature_map.device
        normals = self.compute_gaussian_normals(pred, device=device)  # (B, N, 3)

        # [Question]: Is the view direction always facing same side as the normal?
        view_dir = -centers3d
        view_dir = view_dir / (torch.norm(view_dir, dim=-1, keepdim=True) + 1e-8)
        angle_weight = torch.sum(normals * view_dir, dim=-1).clamp_min(0.0)

        depth_map = self.build_depth_map(pred, img_shape=img_shape, device=device)
        H_img, W_img = int(img_shape[0]), int(img_shape[1])

        x = torch.round(centers2d[..., 0]).to(torch.long)
        y = torch.round(centers2d[..., 1]).to(torch.long)
        x = x.clamp(0, W_img - 1)
        y = y.clamp(0, H_img - 1)

        # [Question]: How to vectorize this depth sampling?
        depth_samples = depth_map[torch.arange(depth_map.shape[0])[:, None], y, x]
        center_depth = centers3d[..., 2]
        visible = center_depth <= (depth_samples + depth_eps)
        visible = visible | torch.isinf(depth_samples)
        visibility = visible.to(dtype=feature_map.dtype)

        return angle_weight * visibility

    def feature_sample(
        self, feature_map, pred, img_shape: tuple = None, centers2d: torch.Tensor = None
    ):
        """Sample per-Gaussian local features from a feature map (batched).

        Returns: (B, N, C)
        """
        if centers2d is None:
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

    def feature_sample_with_visibility(
        self,
        feature_map,
        pred,
        img_shape: tuple = None,
        depth_eps: float = 1e-3,
    ):
        """Sample per-view features and compute view weights for aggregation."""
        centers2d = self.compute_gaussian_coord2d(
            feature_map, pred, img_shape=img_shape
        )
        centers3d = self.compute_gaussian_coord3d(feature_map, pred)

        local_feats = self.feature_sample(
            feature_map, pred, img_shape=img_shape, centers2d=centers2d
        )  # (B, N, C)

        if img_shape is None:
            img_shape = feature_map.shape[-2:]

        view_weights = self.compute_view_weights(
            feature_map,
            pred,
            centers2d=centers2d,
            centers3d=centers3d,
            img_shape=img_shape,
            depth_eps=depth_eps,
        )
        return local_feats, view_weights, centers3d
