# file: scripts/visualize_preds.py
import os
import argparse
import sys
from PIL import Image

# --- Add this block at the top of your script ---
# Define the absolute path to the cloned PaddleOCR repository
# Use 'r' before the string to handle backslashes correctly on Windows.
PADDLE_OCR_REPO_PATH = r'C:\Users\gange\PaddleOCR' # <-- This path must be correct!

# Add the repository's root directory to Python's path
sys.path.append(PADDLE_OCR_REPO_PATH)
# --------------------------------------------------

# Now, the rest of your imports should work as intended
from paddleocr import PaddleOCR
from tools.infer.utility import draw_ocr


def visualize(det_model_dir, rec_model_dir, image_dir, output_dir, font_path):
    """Loads custom models and visualizes predictions."""

    print("Loading custom models... 🧠")
    ocr_engine = PaddleOCR(
    use_textline_orientation=True,
    text_detection_model_dir=det_model_dir,
    text_recognition_model_dir=rec_model_dir
)


    os.makedirs(output_dir, exist_ok=True)
    print(f"Processing images in: {image_dir}")

    for image_name in os.listdir(image_dir):
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        image_path = os.path.join(image_dir, image_name)
        print(f"Inferencing on: {image_name}")
        
        result = ocr_engine.ocr(image_path, cls=True)
        if result and result[0]:
            image = Image.open(image_path).convert('RGB')
            boxes = [line[0] for line in result[0]]
            txts = [line[1][0] for line in result[0]]
            scores = [line[1][1] for line in result[0]]
            
            vis_image = draw_ocr(image, boxes, txts, scores, font_path=font_path)
            save_path = os.path.join(output_dir, f"result_{image_name}")
            Image.fromarray(vis_image).save(save_path)
            print(f"  -> Saved visualization to: {save_path}")

    print("Visualization complete! ✨")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize predictions from custom PaddleOCR models.")
    parser.add_argument('--det_model_dir', type=str, required=True, help='Path to the trained detection model directory.')
    parser.add_argument('--rec_model_dir', type=str, required=True, help='Path to the trained recognition model directory.')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory of test images.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save visualized images.')
    
    # --- IMPROVEMENT: Auto-detect the font path ---
    default_font = os.path.join(PADDLE_OCR_REPO_PATH, 'doc/fonts/latin.ttf')
    parser.add_argument('--font_path', type=str, default=default_font, help='Path to a TTF font file.')

    args = parser.parse_args()
    visualize(args.det_model_dir, args.rec_model_dir, args.image_dir, args.output_dir, args.font_path)