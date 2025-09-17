# file: scripts/coco_to_paddle.py
import json
import os
import argparse
from tqdm import tqdm

def convert_annotations(json_path, image_dir, output_file):
    """Converts COCO-Text V2.0 annotations to PaddleOCR detection format."""
    
    print(f"Loading annotations from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    image_map = {
        img['id']: {
            'path': os.path.join(image_dir, img['file_name']),
            'annotations': []
        }
        for img in data['imgs'].values()
    }

    print("Processing annotations...")
    for ann in tqdm(data['anns'].values()):
        img_id = ann['image_id']
        if img_id in image_map:
            transcription = "###" if ann.get('legibility', 'legible') == 'illegible' else ann['utf8_string']
            
            seg = ann['segmentation'][0]
            points = [[int(seg[i]), int(seg[i+1])] for i in range(0, len(seg), 2)]
            
            if len(points) != 4:
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                min_x, max_x, min_y, max_y = min(x_coords), max(x_coords), min(y_coords), max(y_coords)
                points = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]
            
            annotation = {"transcription": transcription, "points": points}
            image_map[img_id]['annotations'].append(annotation)

    print(f"Writing formatted labels to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for image_info in image_map.values():
            if image_info['annotations']:
                annotations_json = json.dumps(image_info['annotations'], ensure_ascii=False)
                f.write(f"{image_info['path']}\t{annotations_json}\n")
    
    print("Conversion complete! ✅")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert COCO-Text JSON to PaddleOCR format.")
    parser.add_argument('--json_path', type=str, required=True, help='Path to the COCO-Text JSON file.')
    parser.add_argument('--image_dir', type=str, required=True, help='Path to the directory containing images.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the output label file.')
    
    args = parser.parse_args()
    convert_annotations(args.json_path, args.image_dir, args.output_file)
