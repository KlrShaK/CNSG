"""
Run Mask2Former semantic segmentation on NavVis images and save per-pixel class IDs.

Workflow:
1) Load Mask2Former (ADE20K) and its processor.
2) Iterate over session images, run inference, and post-process semantic maps.
3) Save uint8 class ID masks alongside the session.

All paths are resolved from config/paths.yml via config_utils.load_paths().
"""

from typing import List
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))
from utils.config_utils import load_paths

# ================= CONFIGURATION =================
paths = load_paths()
# Path to your RAW images
IMAGE_DIR = paths.raw_images_dir

# Where to save the class ID masks
OUTPUT_DIR = paths.semantic_masks_dir

# Model: ADE20k is best for indoor scenes (150 classes: wall, floor, chair, etc.)
MODEL_ID = "facebook/mask2former-swin-large-ade-semantic"
# =================================================

def main() -> None:
    """
    Run segmentation over all session images and save semantic masks.
    """
    print(f"Loading Model: {MODEL_ID}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    # Load Processor and Model
    processor = Mask2FormerImageProcessor.from_pretrained(MODEL_ID)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL_ID)
    model.to(device)
    model.eval()

    # Setup Output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get List of Images
    valid_exts = {'.jpg', '.jpeg', '.png'}
    image_files: List = [p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in valid_exts]
    image_files.sort()
    
    print(f"Found {len(image_files)} images to process.")

    with torch.no_grad():
        for img_path in tqdm(image_files):
            # 1. Check if already exists (Resume capability)
            save_path = OUTPUT_DIR / f"{img_path.stem}.png"
            
            if save_path.exists():
                continue

            # 2. Load Image
            image = Image.open(img_path).convert("RGB")

            # 3. Inference
            # Mask2Former handles resizing internally, but we pass the original size 
            # to post_process so the output mask matches your high-res input.
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            
            # 4. Post-Process to Semantic Map
            # This returns a tensor of shape (H, W) where values are Class IDs
            target_size = image.size[::-1] # (H, W)
            predicted_semantic_map = processor.post_process_semantic_segmentation(
                outputs, target_sizes=[target_size]
            )[0]
            
            # 5. Save as PNG
            # IDs are 0-150, so uint8 is sufficient and saves space
            mask_np = predicted_semantic_map.cpu().numpy().astype(np.uint8)
            Image.fromarray(mask_np).save(save_path)

    print("Segmentation Complete!")
    print(f"Masks saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
