"""
Camera encoder module for processing camera intrinsics into tokens.
Adapted from Depth-Anything-3's CameraEnc but simplified for intrinsics only.
"""
import torch
import torch.nn as nn
import math

# The descriptor used by every reported model (order matters for old checkpoints).
DEFAULT_CAM_FEATURES = ('fx', 'fy', 'cx', 'cy', 'fovh', 'fovw', 'aspect', 'favg', 'area')


def intrinsics_to_encoding(intrinsics, image_size_hw, features=DEFAULT_CAM_FEATURES):
    """
    Convert camera intrinsics to a compact encoding.
    
    Args:
        intrinsics: Camera intrinsics matrix (B, 3, 3) or (B, 1, 3, 3)
        image_size_hw: Image size as (H, W); each entry is a scalar shared by
            the batch or a (B,) tensor of per-sample sizes (mixed-dataset
            batches, where every sample's K refers to its own image size)

    Returns:
        Encoding tensor of shape (B, 9) containing:
        - fx, fy, cx, cy (4 values)
        - fov_h, fov_w (2 values)
        - aspect_ratio, focal_length_avg (2 values)
        - image_area (1 value)
    """
    # Handle batch dimension
    if intrinsics.dim() == 4:
        intrinsics = intrinsics.squeeze(1)  # (B, 3, 3)

    B = intrinsics.shape[0]
    H, W = (torch.as_tensor(v, device=intrinsics.device, dtype=intrinsics.dtype).expand(B)
            for v in image_size_hw)

    # Extract intrinsic parameters
    fx = intrinsics[:, 0, 0]  # (B,)
    fy = intrinsics[:, 1, 1]  # (B,)
    cx = intrinsics[:, 0, 2]  # (B,)
    cy = intrinsics[:, 1, 2]  # (B,)
    
    # Calculate field of view
    fov_h = 2 * torch.atan((H / 2.0) / torch.clamp(fy, min=1e-6))  # (B,)
    fov_w = 2 * torch.atan((W / 2.0) / torch.clamp(fx, min=1e-6))  # (B,)
    
    # Additional features
    aspect_ratio = W / H  # (B,)
    focal_length_avg = (fx + fy) / 2.0  # (B,)
    image_area = H * W  # (B,)
    
    # Normalize values for better training stability
    # Normalize focal lengths by image size
    fx_norm = fx / W
    fy_norm = fy / H
    cx_norm = cx / W
    cy_norm = cy / H
    
    feats = {
        'fx': fx_norm,
        'fy': fy_norm,
        'cx': cx_norm,
        'cy': cy_norm,
        'fovh': fov_h / math.pi,  # Normalize to [0, 1] range
        'fovw': fov_w / math.pi,
        'aspect': aspect_ratio,
        'favg': focal_length_avg / torch.maximum(H, W),
        'area': image_area / (1920 * 1080),  # Normalize by typical image size
        # extra candidates, used only by the descriptor ablation
        'logfx': torch.log(torch.clamp(fx_norm, min=1e-6)),
        'logfy': torch.log(torch.clamp(fy_norm, min=1e-6)),
        'invfx': 1.0 / torch.clamp(fx_norm, min=1e-6),
        'invfy': 1.0 / torch.clamp(fy_norm, min=1e-6),
    }
    encoding = torch.stack([feats[k] for k in features], dim=-1)  # (B, len(features))

    return encoding


class CameraEncoder(nn.Module):
    """
    Camera encoder that processes camera intrinsics into tokens compatible with DINOv2.
    
    The encoder:
    1. Converts intrinsics to a compact encoding
    2. Projects through MLP to match DINOv2 embedding dimension
    3. Applies transformer blocks for refinement
    4. Outputs tokens that can be injected into DINOv2
    """
    
    def __init__(
        self,
        dim_out: int = 1024,  # DINOv2 embedding dimension (vitl=1024, vitb=768, vits=384, vitg=1536)
        dim_in: int = 9,  # Input encoding dimension
        trunk_depth: int = 4,  # Number of transformer blocks
        num_heads: int = 16,  # Number of attention heads (should match dim_out)
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        features=None,  # descriptor feature names; None = DEFAULT_CAM_FEATURES
        **kwargs,
    ):
        super().__init__()
        self.features = tuple(features) if features else DEFAULT_CAM_FEATURES
        dim_in = len(self.features)
        self.dim_out = dim_out
        self.trunk_depth = trunk_depth
        
        # Project intrinsics encoding to embedding dimension
        self.pose_branch = nn.Sequential(
            nn.Linear(dim_in, dim_out // 2),
            nn.GELU(),
            nn.Linear(dim_out // 2, dim_out),
        )
        
        # Transformer blocks for refinement
        self.trunk = nn.ModuleList([
            self._make_block(dim_out, num_heads, mlp_ratio, init_values)
            for _ in range(trunk_depth)
        ])
        
        self.token_norm = nn.LayerNorm(dim_out)
        self.trunk_norm = nn.LayerNorm(dim_out)

        # Zero-initialized gate so the cam_token starts at exactly 0.
        # This preserves pretrained CLS-token behaviour at the beginning of
        # training and lets the encoder gradually learn to contribute.
        self.output_gate = nn.Parameter(torch.zeros(1))
    
    def _make_block(self, dim, num_heads, mlp_ratio, init_values):
        """Create a transformer block."""
        from .dinov2_layers.block import Block
        from .dinov2_layers.mlp import Mlp
        
        return Block(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            proj_bias=True,
            ffn_bias=True,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            act_layer=nn.GELU,
            ffn_layer=Mlp,
            init_values=init_values,
        )
    
    def forward(self, intrinsics, image_size):
        """
        Process camera intrinsics into tokens.
        
        Args:
            intrinsics: Camera intrinsics (B, 3, 3) or (B, 1, 3, 3)
            image_size: Image size as (H, W) tuple or tensor, or a (B, 2)
                tensor of per-sample [H, W] sizes

        Returns:
            Camera tokens of shape (B, 1, dim_out) - single token per image
        """
        # Convert to encoding
        if isinstance(image_size, tuple):
            H, W = image_size
        elif torch.is_tensor(image_size) and image_size.dim() == 2:
            H, W = image_size[:, 0], image_size[:, 1]
        else:
            H, W = image_size[0], image_size[1]
        
        pose_encoding = intrinsics_to_encoding(intrinsics, (H, W), self.features)  # (B, dim_in)
        
        # Project to embedding dimension
        pose_tokens = self.pose_branch(pose_encoding)  # (B, dim_out)
        pose_tokens = pose_tokens.unsqueeze(1)  # (B, 1, dim_out)
        pose_tokens = self.token_norm(pose_tokens)
        
        # Refine through transformer blocks
        for block in self.trunk:
            pose_tokens = block(pose_tokens)
        
        pose_tokens = self.trunk_norm(pose_tokens)

        # Gate starts at 0 → cam_token ≈ 0 initially, so the pretrained CLS
        # token is unperturbed.  The gate learns to open during training.
        pose_tokens = pose_tokens * self.output_gate

        return pose_tokens  # (B, 1, dim_out)

