# Depth Anything V2 Architecture Sizes

This document describes the architecture dimensions for all encoder types used in Depth Anything V2.

## DINOv2 Encoder Embedding Dimensions

| Encoder  | Embed Dim | Description |
|----------|-----------|-------------|
| **vits** | 384       | ViT-Small   |
| **vitb** | 768       | ViT-Base    |
| **vitl** | 1024      | ViT-Large   |
| **vitg** | 1536      | ViT-Giant   |

## Model Configuration

### ViT-Small (vits)
- **Encoder**: `vits`
- **Embedding Dimension**: 384
- **Features**: 64
- **Out Channels**: [48, 96, 192, 384]
- **Attention Heads**: 6
- **Intermediate Layers**: [2, 5, 8, 11]

### ViT-Base (vitb)
- **Encoder**: `vitb`
- **Embedding Dimension**: 768
- **Features**: 128
- **Out Channels**: [96, 192, 384, 768]
- **Attention Heads**: 12
- **Intermediate Layers**: [2, 5, 8, 11]

### ViT-Large (vitl)
- **Encoder**: `vitl`
- **Embedding Dimension**: 1024
- **Features**: 256
- **Out Channels**: [256, 512, 1024, 1024]
- **Attention Heads**: 16
- **Intermediate Layers**: [4, 11, 17, 23]

### ViT-Giant (vitg)
- **Encoder**: `vitg`
- **Embedding Dimension**: 1536
- **Features**: 384
- **Out Channels**: [1536, 1536, 1536, 1536]
- **Attention Heads**: 24
- **Intermediate Layers**: [9, 19, 29, 39]

## Key Parameter Shapes

### Pretrained Backbone (DINOv2)
- `pretrained.cls_token`: `(1, 1, embed_dim)`
- `pretrained.pos_embed`: `(1, 1370, embed_dim)`
- `pretrained.mask_token`: `(1, embed_dim)`
- `pretrained.patch_embed.proj.weight`: `(embed_dim, 3, 14, 14)`
- `pretrained.patch_embed.proj.bias`: `(embed_dim,)`
- `pretrained.blocks.{i}.norm1.weight`: `(embed_dim,)`
- `pretrained.blocks.{i}.norm1.bias`: `(embed_dim,)`
- `pretrained.blocks.{i}.attn.qkv.weight`: `(embed_dim * 3, embed_dim)`
- `pretrained.blocks.{i}.attn.qkv.bias`: `(embed_dim * 3,)`
- `pretrained.blocks.{i}.attn.proj.weight`: `(embed_dim, embed_dim)`
- `pretrained.blocks.{i}.attn.proj.bias`: `(embed_dim,)`

### Camera Encoder (if enabled)
- `cam_encoder.pose_branch.*`: Depends on `embed_dim`
- `cam_encoder.trunk.*`: Depends on `embed_dim` and `num_heads`

### Depth Head
- Depends on `features` and `out_channels` configuration

## Encoder Detection from Checkpoint

To detect the encoder type from a checkpoint, check the shape of `pretrained.patch_embed.proj.weight`:
- Shape `(384, 3, 14, 14)` → **vits**
- Shape `(768, 3, 14, 14)` → **vitb**
- Shape `(1024, 3, 14, 14)` → **vitl**
- Shape `(1536, 3, 14, 14)` → **vitg**

Or check `pretrained.cls_token`:
- Shape `(1, 1, 384)` → **vits**
- Shape `(1, 1, 768)` → **vitb**
- Shape `(1, 1, 1024)` → **vitl**
- Shape `(1, 1, 1536)` → **vitg**

