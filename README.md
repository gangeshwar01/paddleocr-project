# End-to-End PaddleOCR Training Project

This repository provides a complete workflow for training custom text detection and recognition models using the PaddleOCR framework. It includes data preparation scripts, training notebooks for Google Colab, and visualization utilities.

[Image of PaddleOCR architecture diagram]

## 🚀 Project Overview

The goal is to demonstrate an end-to-end OCR pipeline:
1.  **Data Preparation**: Convert standard datasets (like COCO-Text) into PaddleOCR-compatible formats.
2.  **Training**: Fine-tune detection (DBNet) and recognition (CRNN/SVTR) models on a custom dataset.
3.  **Evaluation**: Measure model performance using standard OCR metrics.
4.  **Inference**: Use the trained models to perform OCR and visualize the results.

## 📁 Repository Structure

```
paddleocr-project/
├─ notebooks/         # Colab notebooks for training
│  ├─ colab_train_detector.ipynb
│  └─ colab_train_recognizer.ipynb
├─ scripts/           # Standalone Python scripts
│  ├─ coco_to_paddle.py
│  ├─ prepare_rec.py
│  └─ visualize_preds.py
├─ configs/           # Your modified training configs
├─ outputs/           # Trained models, logs, and results
├─ docs/              # Detailed documentation
│  ├─ report.pdf       (Generated from a Markdown/LaTeX source)
│  └─ architecture_diagram.drawio.svg
└─ README.md
```

## ⚙️ How to Use

### 1. Setup

First, clone the official PaddleOCR repository, as our scripts and notebooks rely on its tools.

```bash
git clone [https://github.com/PaddlePaddle/PaddleOCR.git](https://github.com/PaddlePaddle/PaddleOCR.git)
cd PaddleOCR
pip install -r requirements.txt
```
Place the contents of this project (`scripts`, `notebooks`, etc.) inside the cloned `PaddleOCR` directory to ensure all paths work correctly.

### 2. Prepare Your Dataset

-   **For Detection**: Use `coco_to_paddle.py` to convert COCO-Text or similar JSON annotations.
-   **For Recognition**: Use `prepare_rec.py` to create label files from a directory of cropped word images.

```bash
# Example for detection data
python3 ../scripts/coco_to_paddle.py --json_path /path/to/cocotext.json --image_dir /path/to/images --output_file data/det_train.txt

# Example for recognition data
python3 ../scripts/prepare_rec.py --image_dir /path/to/crops --gt_file /path/to/gt.txt --output_dir data/
```

### 3. Train a Model

Open one of the notebooks in `notebooks/` in Google Colab or Kaggle. They provide a step-by-step guide to:
- Install dependencies.
- Download and prepare a dataset.
- Configure and launch training.
- Evaluate the trained model.

### 4. Visualize Predictions

After training, use the `visualize_preds.py` script to see your model in action.

```bash
python3 ../scripts/visualize_preds.py \
    --det_model_dir ./output/my_det_model/best_accuracy \
    --rec_model_dir ./output/my_rec_model/best_accuracy \
    --image_dir ./data/test_images \
    --output_dir ./output/visualizations
```

For a deep dive into the architecture and methodology, please see the full technical report in `docs/report.pdf`.
