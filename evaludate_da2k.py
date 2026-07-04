import cv2
import torch
import json
import os
from tqdm import tqdm
from depth_anything_v2.dpt import DepthAnythingV2

def evaluate_da2k(annotation_path, image_dir, model_path):
    # 1. Initialize the DINOv2-based Large model
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_configs = {
        'encoder': 'vitl', 
        'features': 256, 
        'out_channels': [256, 512, 1024, 1024]
    }
    
    print(f"Loading Large model on {DEVICE}...")
    model = DepthAnythingV2(**model_configs)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model = model.to(DEVICE).eval()

    # 2. Load DA-2K annotations
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)

    correct_pairs = 0
    total_pairs = 0

    print("Starting evaluation...")
    with torch.no_grad():
        for img_path, points in tqdm(annotations.items()):
            full_img_path = os.path.join(image_dir, img_path)
            raw_img = cv2.imread(full_img_path)
            
            if raw_img is None:
                print(f"Warning: Could not read {full_img_path}")
                continue
                
            # Predict depth map
            depth = model.infer_image(raw_img) 
            
            # The model outputs inverse depth (disparity), so larger values = closer to camera
            for pair in points:
                h1, w1 = pair['point1']
                h2, w2 = pair['point2']
                gt_closer = pair['closer_point']
                
                # Ensure coordinates are within bounds
                h1, w1 = min(h1, depth.shape[0]-1), min(w1, depth.shape[1]-1)
                h2, w2 = min(h2, depth.shape[0]-1), min(w2, depth.shape[1]-1)
                
                d1 = depth[h1, w1]
                d2 = depth[h2, w2]
                
                # Predict which point is closer
                pred_closer = 'point1' if d1 > d2 else 'point2'
                
                if pred_closer == gt_closer:
                    correct_pairs += 1
                total_pairs += 1

    accuracy = (correct_pairs / total_pairs) * 100 if total_pairs > 0 else 0
    print(f"\n--- Evaluation Complete ---")
    print(f"Total pairs evaluated: {total_pairs}")
    print(f"DA-2K Accuracy: {accuracy:.2f}%")

if __name__ == '__main__':
    # Update these paths to match your local setup
    ANNOTATION_PATH = 'DA-2K/annotations.json'
    IMAGE_DIR = 'DA-2K/images'
    MODEL_PATH = 'models/raw_models/DepthAnythingV2/checkpoints/depth_anything_v2_vitl.pth   '
    
    evaluate_da2k(ANNOTATION_PATH, IMAGE_DIR, MODEL_PATH)
