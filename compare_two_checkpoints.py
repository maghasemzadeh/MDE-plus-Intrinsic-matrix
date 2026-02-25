#!/usr/bin/env python3
"""
Compare two da2-revised checkpoints to find why one works and the other outputs zeros.

Usage:
  python compare_two_checkpoints.py <working_checkpoint> <broken_checkpoint>

Example (run on server via SSH, use absolute paths):
  cd /home/ubuntu/projects/MDE-plus-Intrinsic-matrix
  source venv/bin/activate
  python compare_two_checkpoints.py \
    /home/ubuntu/projects/MDE-plus-Intrinsic-matrix/models/raw_models/DepthAnythingV2-revised/checkpoints/metric_finetuning/revised_basic_vitl_nyu+middlebury_intrinsics_lr5e-6_bs4_20260224_104321/best.pth \
    /home/ubuntu/projects/MDE-plus-Intrinsic-matrix/models/raw_models/DepthAnythingV2-revised/checkpoints/basic_finetuning/revised_basic_vitl_nyu+vkitti+diode_intrinsics_lr5e-6_bs4_20260224_110026/best.pth
"""
import os
import sys
import numpy as np
import torch

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def resolve_path(p: str) -> str:
    """Try .pth if .pt given and file not found."""
    if os.path.exists(p):
        return p
    alt = p.replace(".pt", ".pth") if p.endswith(".pt") else p.replace(".pth", ".pt")
    if os.path.exists(alt):
        return alt
    return p


def inspect_ckpt(path: str) -> dict:
    """Load checkpoint and return key info."""
    path = resolve_path(path)
    if not os.path.exists(path):
        return {"error": f"File not found: {path}", "path": path}

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    ckpt_keys = set(state_dict.keys())
    depth_head_ckpt = [k for k in ckpt_keys if "depth_head" in k]
    cam_encoder_ckpt = [k for k in ckpt_keys if "cam_encoder" in k]

    return {
        "path": path,
        "config": config,
        "n_keys": len(ckpt_keys),
        "depth_head_keys": len(depth_head_ckpt),
        "cam_encoder_keys": len(cam_encoder_ckpt),
        "state_dict": state_dict,
        "ckpt_keys": ckpt_keys,
        "depth_head_sample": depth_head_ckpt[:3] if depth_head_ckpt else [],
    }


def test_inference(path: str, label: str) -> dict:
    """Run inference and return stats."""
    path = resolve_path(path)
    if not os.path.exists(path):
        return {"error": f"File not found: {path}", "n_positive": 0}

    from models.utils import identify_model_from_checkpoint
    from models import create_model_wrapper

    config = identify_model_from_checkpoint(path)
    encoder = config.get("encoder", "vitl")
    use_intrinsics = config.get("use_camera_intrinsics", False)

    try:
        wrapper = create_model_wrapper("da2-revised", {
            "model_type": "basic",
            "encoder": encoder,
            "checkpoint_path": path,
            "use_camera_intrinsics": use_intrinsics,
        })
    except Exception as e:
        return {"error": str(e), "n_positive": 0}

    dummy_img = np.random.randint(0, 255, (375, 1242, 3), dtype=np.uint8)
    K = np.array([[725, 0, 621], [0, 725, 187], [0, 0, 1]], dtype=np.float32)

    with torch.no_grad():
        depth_noK = wrapper.infer_image(dummy_img, input_size=518, intrinsics=None)
        depth_withK = wrapper.infer_image(dummy_img, input_size=518, intrinsics=K) if use_intrinsics else None

    n_noK = int(np.sum(np.isfinite(depth_noK) & (depth_noK > 0)))
    n_withK = int(np.sum(np.isfinite(depth_withK) & (depth_withK > 0))) if depth_withK is not None else -1

    return {
        "label": label,
        "config": config,
        "n_positive_no_intrinsics": n_noK,
        "n_positive_with_intrinsics": n_withK,
        "depth_range_noK": (float(np.min(depth_noK)), float(np.max(depth_noK))) if depth_noK is not None else None,
        "depth_range_withK": (float(np.min(depth_withK)), float(np.max(depth_withK))) if depth_withK is not None else None,
    }


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    working_path = sys.argv[1]
    broken_path = sys.argv[2]

    print("=" * 70)
    print("COMPARING CHECKPOINTS")
    print("=" * 70)
    print(f"\nWorking: {working_path}")
    print(f"Broken:  {broken_path}\n")

    # 1. Inspect both
    info_working = inspect_ckpt(working_path)
    info_broken = inspect_ckpt(broken_path)

    if "error" in info_working:
        print(f"❌ Working checkpoint: {info_working['error']}")
    else:
        print("WORKING checkpoint:")
        print(f"  Config: {info_working.get('config', {})}")
        print(f"  Keys: {info_working['n_keys']}, depth_head: {info_working['depth_head_keys']}, cam_encoder: {info_working['cam_encoder_keys']}")
        print(f"  depth_head sample: {info_working.get('depth_head_sample', [])}")

    if "error" in info_broken:
        print(f"\n❌ Broken checkpoint: {info_broken['error']}")
    else:
        print("\nBROKEN checkpoint:")
        print(f"  Config: {info_broken.get('config', {})}")
        print(f"  Keys: {info_broken['n_keys']}, depth_head: {info_broken['depth_head_keys']}, cam_encoder: {info_broken['cam_encoder_keys']}")
        print(f"  depth_head sample: {info_broken.get('depth_head_sample', [])}")

    # 2. Key diff
    if "ckpt_keys" in info_working and "ckpt_keys" in info_broken:
        only_working = info_working["ckpt_keys"] - info_broken["ckpt_keys"]
        only_broken = info_broken["ckpt_keys"] - info_working["ckpt_keys"]
        if only_working:
            print(f"\n  Keys only in WORKING: {len(only_working)}")
            for k in sorted(only_working)[:5]:
                print(f"    {k}")
        if only_broken:
            print(f"\n  Keys only in BROKEN: {len(only_broken)}")
            for k in sorted(only_broken)[:5]:
                print(f"    {k}")

    # 3. Test inference
    print("\n" + "=" * 70)
    print("INFERENCE TEST")
    print("=" * 70)

    res_working = test_inference(working_path, "working")
    res_broken = test_inference(broken_path, "broken")

    if "error" in res_working:
        print(f"\n❌ Working: {res_working['error']}")
    else:
        print(f"\n✅ Working: n_>0 (no K)={res_working['n_positive_no_intrinsics']}, (with K)={res_working['n_positive_with_intrinsics']}")
        print(f"   depth range noK: {res_working['depth_range_noK']}")

    if "error" in res_broken:
        print(f"\n❌ Broken: {res_broken['error']}")
    else:
        ok = res_broken["n_positive_no_intrinsics"] > 0 or res_broken["n_positive_with_intrinsics"] > 0
        sym = "✅" if ok else "❌"
        print(f"\n{sym} Broken:  n_>0 (no K)={res_broken['n_positive_no_intrinsics']}, (with K)={res_broken['n_positive_with_intrinsics']}")
        print(f"   depth range noK: {res_broken['depth_range_noK']}")
        if not ok:
            print("\n   => Broken checkpoint produces ZERO positive depth values.")
            print("   => Likely cause: depth_head keys not loading (key name mismatch with model)")


if __name__ == "__main__":
    main()
