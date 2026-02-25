#!/usr/bin/env python3
"""
Inspect checkpoint keys vs model keys to debug loading issues.
Run: python inspect_checkpoint_keys.py path/to/best.pth
"""
import os
import sys
import torch

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_checkpoint_keys.py <checkpoint_path>")
        sys.exit(1)
    
    ckpt_path = sys.argv[1]
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)
    
    print(f"Inspecting: {ckpt_path}\n")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    if isinstance(ckpt, dict) and 'model' in ckpt:
        state_dict = ckpt['model']
        print("Checkpoint format: training (has 'model' key)")
        if 'config' in ckpt:
            print(f"Config: {ckpt['config']}")
    else:
        state_dict = ckpt
        print("Checkpoint format: raw state dict")
    
    ckpt_keys = sorted(state_dict.keys())
    print(f"\nCheckpoint has {len(ckpt_keys)} keys.")
    print("First 15 checkpoint keys:")
    for k in ckpt_keys[:15]:
        print(f"  {k}: {state_dict[k].shape}")
    
    # Now build the model and compare
    print("\n" + "="*60)
    print("Building model (basic, vitl, use_camera_intrinsics from config)...")
    
    # Use the model wrapper to match actual inference path
    from models.utils import identify_model_from_checkpoint
    config = identify_model_from_checkpoint(ckpt_path)
    use_intrinsics = config.get('use_camera_intrinsics', False)
    encoder = config.get('encoder', 'vitl')
    
    revised_path = os.path.join(project_root, 'models', 'raw_models', 'DepthAnythingV2-revised')
    if revised_path not in sys.path:
        sys.path.insert(0, revised_path)
    
    from depth_anything_v2.dpt import DepthAnythingV2
    model = DepthAnythingV2(encoder=encoder, use_camera_intrinsics=use_intrinsics)
    model_keys = sorted(model.state_dict().keys())
    
    print(f"Model has {len(model_keys)} keys.")
    print("First 15 model keys:")
    for k in model_keys[:15]:
        print(f"  {k}: {model.state_dict()[k].shape}")
    
    # Compare
    print("\n" + "="*60)
    ckpt_set = set(ckpt_keys)
    model_set = set(model_keys)
    
    # Strip prefix for comparison
    def strip(k):
        for p in ('model.', 'module.'):
            if k.startswith(p):
                return k[len(p):]
        return k
    ckpt_stripped = {strip(k): k for k in ckpt_keys}
    ckpt_stripped_set = set(ckpt_stripped.keys())
    
    in_both = model_set & ckpt_stripped_set
    only_model = model_set - ckpt_stripped_set
    only_ckpt = ckpt_stripped_set - model_set
    
    print(f"Keys in BOTH (will load): {len(in_both)}")
    print(f"Keys ONLY in model (missing from ckpt): {len(only_model)}")
    if only_model:
        for k in sorted(only_model)[:10]:
            print(f"  MISSING: {k}")
        if len(only_model) > 10:
            print(f"  ... and {len(only_model)-10} more")
    
    print(f"\nKeys ONLY in checkpoint (unexpected): {len(only_ckpt)}")
    if only_ckpt:
        for k in sorted(only_ckpt)[:10]:
            print(f"  UNEXPECTED: {k} (ckpt key: {ckpt_stripped.get(k, '?')})")
        if len(only_ckpt) > 10:
            print(f"  ... and {len(only_ckpt)-10} more")
    
    # Check depth_head specifically
    depth_head_ckpt = [k for k in ckpt_keys if 'depth_head' in k]
    depth_head_model = [k for k in model_keys if 'depth_head' in k]
    print(f"\nDepth head keys - ckpt: {len(depth_head_ckpt)}, model: {len(depth_head_model)}")
    if depth_head_ckpt:
        print("  Sample ckpt depth_head keys:", depth_head_ckpt[:3])
    if depth_head_model:
        print("  Sample model depth_head keys:", depth_head_model[:3])

if __name__ == '__main__':
    main()
