import torch
import torch.nn as nn
import torch.nn.functional as F
from avatar_utils.config import load_config


class GaussianDecoder(nn.Module):
    """
    MLP head to predict Gaussian parameters taking in the encoding of pose info + appearance features,
    while incorporating identity latent code as a FiLM layer to modulate the features.

    Output parameterization (per Gaussian):
      - scales: 3 values (positive radii)
      - rotation: quaternion (4)
      - alpha: opacity (1)
      - sh: K spherical-harmonics coeffs (remaining dims)
    """

    def __init__(self):
        super().__init__()
        cfg = load_config() or {}
        dec_cfg = cfg.get("decoder", {})

        self.in_dim = int(dec_cfg.get("in_dim", 515))
        self.hidden = int(dec_cfg.get("hidden", 256))
        self.out_dim = int(dec_cfg.get("out_dim", 11))
        self.z_dim = int(cfg.get("identity_encoder", {}).get("latent_dim", 64))

        # first local block
        self.fc1 = nn.Linear(self.in_dim, self.hidden)
        self.activation1 = nn.ReLU(inplace=True)

        # FiLM film_net
        self.film_net = nn.Linear(self.z_dim, 2 * self.hidden)

        # remaining MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden, self.out_dim),
        )

    def forward(self, combined_feats, z_id):
        """
        combined_feats: (B, N, in_dim)
        z_id: (B, z_dim) if FiLM is used (z_dim must be set in config)

        Returns a dict of parameterized Gaussian fields (batched):
          scales: (B,N,3), rotation: (B,N,4), alpha: (B,N,1), sh: (B,N,K)
        """
        # Support chunked decoding over the Gaussian dimension to reduce peak VRAM
        cfg = load_config() or {}
        dec_cfg = cfg.get("decoder", {})
        chunk_size = int(dec_cfg.get("chunk_size", 8192))

        B, N, _ = combined_feats.shape

        assert (
            z_id is not None
        ), "z_id must be provided when deIcoder configured with z_dim"

        def _fuse_batched_chunk(batched):
            # Same logic as fuse_batch, but operates on a chunk and returns per-Gaussian tensors
            scales = batched["scales"]
            quats = batched["rotation"]
            alpha = batched["alpha"]
            sh = batched.get("sh", None)

            if alpha.dim() == 3 and alpha.shape[-1] == 1:
                alpha = alpha.squeeze(-1)

            w = alpha.clamp(0.0, 1.0)
            w_sum = w.sum(dim=0, keepdim=True).clamp_min(1e-8)
            w = w / w_sum

            scales_agg = (w.unsqueeze(-1) * scales).sum(dim=0)
            sh_agg = (
                (w.unsqueeze(-1) * sh).sum(dim=0)
                if (sh is not None and sh.numel() > 0)
                else None
            )

            qqT = quats.unsqueeze(-1) @ quats.unsqueeze(-2)  # (B,nc,4,4)
            M = (w.unsqueeze(-1).unsqueeze(-1) * qqT).sum(dim=0)  # (nc,4,4)
            _, eigvecs = torch.linalg.eigh(M)
            avg_q = eigvecs[..., -1]
            avg_q = avg_q / (avg_q.norm(dim=-1, keepdim=True) + 1e-12)

            alpha_mean = (w * alpha).sum(dim=0)

            return {
                "scales": scales_agg,
                "rotation": avg_q,
                "alpha": alpha_mean,
                "sh": sh_agg,
            }

        # Precompute FiLM gamma/beta once per batch and reuse for chunks
        gamma_beta = self.film_net(z_id)  # (B, 2H)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1)  # (B,1,H)
        beta = beta.unsqueeze(1)  # (B,1,H)

        fused_parts = {"scales": [], "rotation": [], "alpha": [], "sh": []}
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            feats_chunk = combined_feats[:, start:end, :]  # (B,nc,in_dim)

            # First block + FiLM
            h = self.fc1(feats_chunk)  # (B,nc,H)
            h = (1.0 + gamma) * h + beta
            h = self.activation1(h)

            # Remaining MLP
            out = self.mlp(h)  # (B,nc,out_dim)

            # Parameterize and fuse across batch
            split_out = self.split_and_parameterize(out)
            fused_chunk = _fuse_batched_chunk(split_out)  # per-gaussian within chunk

            fused_parts["scales"].append(fused_chunk["scales"])  # (nc,3)
            fused_parts["rotation"].append(fused_chunk["rotation"])  # (nc,4)
            fused_parts["alpha"].append(fused_chunk["alpha"])  # (nc,) or (nc,1)
            if fused_chunk["sh"] is not None:
                fused_parts["sh"].append(fused_chunk["sh"])  # (nc,K)
            else:
                # Keep alignment: append None marker; we'll handle after the loop
                fused_parts["sh"].append(None)

            # Free chunk temporaries ASAP
            del feats_chunk, h, out, split_out, fused_chunk
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Concatenate chunk results
        scales = torch.cat([x for x in fused_parts["scales"]], dim=0)
        rotation = torch.cat([x for x in fused_parts["rotation"]], dim=0)
        alpha_list = fused_parts["alpha"]
        alpha = torch.cat(
            [x if x.dim() == 1 else x.squeeze(-1) for x in alpha_list], dim=0
        )

        # If any chunk had SH, stack; else set to None
        if any(x is not None for x in fused_parts["sh"]):
            sh = torch.cat([x for x in fused_parts["sh"] if x is not None], dim=0)
        else:
            sh = None

        return {"scales": scales, "rotation": rotation, "alpha": alpha, "sh": sh}

    def split_and_parameterize(self, out):
        """
        Split raw MLP outputs into parameter fields and apply stable parameterizations.
        """
        D = out.shape[-1]
        min_header = 3 + 4 + 1
        if D < min_header:
            raise ValueError(f"Output dim must be >= {min_header}, got {D}")

        sh_dim = D - min_header
        i = 0
        scales_raw = out[..., i : i + 3]
        i += 3
        rot_raw = out[..., i : i + 4]
        i += 4
        alpha_raw = out[..., i : i + 1]
        i += 1
        sh_raw = (
            out[..., i : i + sh_dim]
            if sh_dim > 0
            else out.new_zeros((*out.shape[:-1], 0))
        )

        eps = 1e-6
        scales = F.softplus(scales_raw) + eps
        scales = torch.clamp(scales, min=1e-6, max=3.0)

        rot_norm = torch.linalg.norm(rot_raw, dim=-1, keepdim=True)
        rot = rot_raw / (rot_norm + 1e-8)

        alpha = torch.sigmoid(alpha_raw)
        alpha = torch.clamp(alpha, min=1e-6, max=1.0)

        sh = torch.tanh(sh_raw) * 0.5

        return {
            "scales": scales,
            "rotation": rot,
            "alpha": alpha,
            "sh": sh,
        }

    def fuse_batch(self, param_dics):
        """
        Fuse a batched dict of gaussian params into one per-Gaussian set.

        Expects a dict with tensors shaped like:
          - scales: (B, N, 3)
          - rotation: (B, N, 4)  (quaternions)
          - alpha: (B, N) or (B, N, 1)
          - sh: optional (B, N, K)

        Returns dict with per-Gaussian tensors (N, ...): scales, rotation, alpha_mean,
        alpha_union, sh.
        """
        batched = param_dics
        scales = batched["scales"]
        quats = batched["rotation"]
        alpha = batched["alpha"]
        sh = batched.get("sh", None)

        if alpha.dim() == 3 and alpha.shape[-1] == 1:
            alpha = alpha.squeeze(-1)

        w = alpha.clamp(0.0, 1.0)
        w_sum = w.sum(dim=0, keepdim=True).clamp_min(1e-8)
        w = w / w_sum

        scales_agg = (w.unsqueeze(-1) * scales).sum(dim=0)
        sh_agg = (
            (w.unsqueeze(-1) * sh).sum(dim=0)
            if (sh is not None and sh.numel() > 0)
            else None
        )

        # Quaternion averaging via eigen decomposition
        qqT = quats.unsqueeze(-1) @ quats.unsqueeze(-2)  # (B,N,4,4)
        M = (w.unsqueeze(-1).unsqueeze(-1) * qqT).sum(dim=0)  # (N,4,4)
        _, eigvecs = torch.linalg.eigh(M)
        avg_q = eigvecs[..., -1]
        avg_q = avg_q / (avg_q.norm(dim=-1, keepdim=True) + 1e-12)

        alpha_mean = (w * alpha).sum(dim=0)
        alpha_union = 1.0 - torch.prod(1.0 - alpha.clamp(0.0, 1.0), dim=0)

        return {
            "scales": scales_agg,  # (N, 3)
            "rotation": avg_q,
            "alpha": alpha_mean,
            # "alpha_mean": alpha_mean,
            # "alpha_union": alpha_union,
            "sh": sh_agg,
        }
