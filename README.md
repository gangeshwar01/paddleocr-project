# A Comprehensive Study of Optical Character Recognition (OCR)

This repository provides a complete, hands-on study of Optical Character Recognition systems. It covers a production-grade framework (**PaddleOCR**), from-scratch implementations of foundational papers (**PyTorch**), and a classic alternative object detector (**YOLOv3/Darknet**).

The primary goal is to offer a full-circle learning experience, from understanding the core architectures to training and deploying custom OCR models.

## 📂 Repository Structure

```
.
├── PaddleOCR/
│   ├── README.md
│   └── notebooks/
│       ├── 1_Data_Preparation.ipynb
│       ├── 2_Detection_Training.ipynb
│       └── 3_Recognition_Training.ipynb
│
├── Paper Implementation with Pytorch/
│   ├── README.md
│   ├── dbnet/
│   └── crnn/
│
└── Darknet and YOLO/
    ├── README.md
    ├── yolov3-custom.cfg
    └── ...
```

  - **`PaddleOCR/`**: The main focus, containing a full, practical workflow for training custom models with PaddleOCR.
  - **`Paper Implementation with Pytorch/`**: Educational, from-scratch implementations of the core DBNet and CRNN architectures.
  - **`Darknet and YOLO/`**: A guide to training a YOLOv3 object detector using the Darknet framework, serving as a reference for a classic, single-stage detection approach.

-----

## 🛠️ Core Architecture: The PP-OCR Pipeline

The PP-OCR system is an efficient, modular pipeline designed for practical use. It consists of three main stages that process an image to extract text.

**Pipeline Flow:** `Image → Text Detector → [Optional] Angle Classifier → Text Recognizer → Final Text`

### 1\. Text Detector (DBNet)

  - **Purpose**: To locate all text instances in an image and draw tight bounding polygons around them.
  - **Architecture**: Based on **DBNet (Differentiable Binarization)**.
      - **Backbone**: A feature extractor like **MobileNetV3** (for lightweight models) or **ResNet** (for server models).
      - **Neck**: A Feature Pyramid Network (FPN) to fuse multi-scale features, enabling detection of both small and large text.
      - **Head**: Outputs a probability map and a threshold map, which are combined to produce clean, accurate bounding boxes, even for curved or irregular text.

### 2\. Angle Classifier

  - **Purpose**: To correct the orientation of a detected text box.
  - **Architecture**: A small, fast CNN-based image classifier. It predicts if a text crop is upright (0 degrees) or upside-down (180 degrees) and rotates it if necessary before recognition. This is a simple step that significantly improves recognition accuracy.

### 3\. Text Recognizer (CRNN / SVTR)

  - **Purpose**: To transcribe the character sequence from the cropped and corrected text image.
  - **Architecture**: The default is **CRNN (Convolutional Recurrent Neural Network)**.
      - **Backbone (CNN)**: Extracts visual features from the text image.
      - **Neck (RNN)**: A Bi-LSTM layer that treats the features as a sequence, capturing contextual information between characters.
      - **Head (CTC)**: A Connectionist Temporal Classification (CTC) decoder that converts the per-frame predictions from the RNN into the final text label.
      - Newer versions like **SVTR (Scene Vision Transformer)** are also used for higher accuracy by replacing the RNN with attention-based mechanisms.

-----

## 💾 Datasets and Label Formats

The choice of dataset profoundly impacts model performance and generalization.

### Label Formats

**1. Detection Label Format (`det_label.txt`)**

  - Each line corresponds to one image.

  - Format: `image_path\tjson_string`

  - The `json_string` is a list of dictionaries, each containing the text `transcription` and the bounding `points`.

    ```
    images/img_01.png	[{"transcription": "PADDLE", "points": [[10, 15], [100, 16], [99, 50], [11, 49]]}, {"transcription": "OCR", "points": [[120, 20], [180, 20], [180, 55], [120, 55]]}]
    ```

**2. Recognition Label Format (`rec_label.txt`)**

  - Each line maps one cropped word image to its label.

  - Format: `image_path\tlabel`

    ```
    word_crops/crop_0001.png	PADDLE
    word_crops/crop_0002.png	OCR
    ```

### Dataset Trade-offs

| Dataset | Pros ✔️ | Cons ❌ | Best For... |
| :--- | :--- | :--- | :--- |
| **COCO-Text v2.0** | Huge scale (63k images), complex real-world scenes, provides masks for segmentation. | Primarily English, annotations can be noisy. | Training highly robust, general-purpose English detectors. |
| **ICDAR 2019 MLT**| Excellent multilingual support (10 languages), benchmark standard for robust reading. | Smaller scale, less scene diversity than COCO-Text. | Building models that need to support multiple languages and scripts. |
| **SynthText** | Massive synthetic dataset (\>800k images), perfect for pre-training models from scratch. | Synthetic data lacks real-world lighting/texture variations, can lead to domain gap. | Pre-training recognition models before fine-tuning on real data. |

**Note on Generalization**: Training on a diverse mix of datasets is key. Pre-training on synthetic data and fine-tuning on a combination of real-world datasets like COCO-Text and ICDAR often yields the most robust models.

-----

## 📈 Training and Benchmarking

### Training Hyperparameters

All training parameters are controlled via **`.yml` configuration files** located in the `PaddleOCR/configs/` directory. Instead of hardcoding, you modify these files to run experiments.

**Key Parameters to Tune:**

  - **`Global.epoch_num`**: Total number of training epochs.
  - **`Optimizer.lr.learning_rate`**: The learning rate.
  - **`Train.loader.batch_size_per_card`**: Batch size per GPU.
  - **`Train.dataset.transforms`**: The data augmentation pipeline (e.g., rotation, perspective, color jitter).
  - **`Global.pretrained_model`**: Path to pre-trained backbone weights to accelerate convergence.

### Sample Benchmark Results

After running your training jobs, you should record your results to compare different models and configurations.

**Detection Performance (on ICDAR 2015 validation set)**
| Model | Backbone | Precision | Recall | F-Measure (H-mean) |
| :--- | :--- | :---: | :---: | :---: |
| DBNet | MobileNetV3 | 0.885 | 0.832 | 0.858 |
| DBNet | ResNet-50 | 0.912 | 0.855 | 0.883 |

**Recognition Performance (on custom validation set)**
| Model | Backbone | Accuracy |
| :--- | :--- | :---: |
| CRNN | MobileNetV3 | 0.921 |
| SVTR | LCNet | 0.945 |

-----

## 🚀 How to Run the Code

### 1\. Environment Setup

Clone this repository and install the necessary dependencies for the section you wish to explore. Each top-level directory contains its own `README.md` with specific setup instructions.

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2\. Run the Full PaddleOCR Workflow

For a complete, end-to-end experience, navigate to the `PaddleOCR` directory and follow the Jupyter notebooks in order.

```bash
cd PaddleOCR/
```

1.  **Prepare Data**: Run `notebooks/1_Data_Preparation.ipynb` to convert your dataset into the required formats.
2.  **Train Detector**: Run `notebooks/2_Detection_Training.ipynb` to train a text detection model.
3.  **Train Recognizer**: Run `notebooks/3_Recognition_Training.ipynb` to train a text recognition model on the cropped words.
