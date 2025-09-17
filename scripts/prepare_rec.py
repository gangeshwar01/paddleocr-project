# file: scripts/prepare_rec.py
import os
import random
import argparse

def create_rec_splits(image_dir, gt_file, output_dir, train_split_ratio=0.9):
    """Creates train/validation label files for recognition training."""

    print(f"Reading ground truth from: {gt_file}")
    with open(gt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    data = []
    for line in lines:
        parts = line.strip().replace(',', '\t', 1).split('\t')
        if len(parts) == 2:
            img_name, transcription = parts
            img_path = os.path.join(image_dir, img_name)
            if os.path.exists(img_path):
                data.append(f"{img_path}\t{transcription}\n")

    random.shuffle(data)
    split_index = int(len(data) * train_split_ratio)
    train_data, val_data = data[:split_index], data[split_index:]

    print(f"Total: {len(data)} | Training: {len(train_data)} | Validation: {len(val_data)}")

    os.makedirs(output_dir, exist_ok=True)
    
    train_label_path = os.path.join(output_dir, 'train_label.txt')
    with open(train_label_path, 'w', encoding='utf-8') as f:
        f.writelines(train_data)
    print(f"Train labels saved to: {train_label_path}")

    val_label_path = os.path.join(output_dir, 'val_label.txt')
    with open(val_label_path, 'w', encoding='utf-8') as f:
        f.writelines(val_data)
    print(f"Validation labels saved to: {val_label_path}")
    print("Splitting complete! ✅")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create recognition data splits.")
    parser.add_argument('--image_dir', type=str, required=True, help='Directory with cropped word images.')
    parser.add_argument('--gt_file', type=str, required=True, help='Ground truth file (image_name,transcription).')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save label files.')
    parser.add_argument('--split_ratio', type=float, default=0.9, help='Training split ratio.')

    args = parser.parse_args()
    create_rec_splits(args.image_dir, args.gt_file, args.output_dir, args.split_ratio)
