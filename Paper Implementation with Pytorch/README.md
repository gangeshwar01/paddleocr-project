# PyTorch Implementations of Foundational OCR Architectures

## 🎯 Objective
This directory contains from-scratch PyTorch implementations of the core algorithms used in modern OCR systems like PaddleOCR. The goal is to provide a clear, educational resource for understanding these models, focusing on architectural correctness and readability over state-of-the-art performance.

---

## 📂 Project Structure
- **/dbnet**: Implementation of **DBNet** for scene text detection.
- **/crnn**: Implementation of **CRNN** with a CTC loss for scene text recognition.

---

## ⚙️ Setup
First, create a virtual environment and install the required dependencies.

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# Install all necessary packages
pip install -r requirements.txt

## Of course. Here is the complete, well-commented code content for all the files in the Paper Implementation with Pytorch directory.

Top-Level Files
README.md
Markdown

# PyTorch Implementations of Foundational OCR Architectures

## 🎯 Objective
This directory contains from-scratch PyTorch implementations of the core algorithms used in modern OCR systems like PaddleOCR. The goal is to provide a clear, educational resource for understanding these models, focusing on architectural correctness and readability over state-of-the-art performance.

---

## 📂 Project Structure
- **/dbnet**: Implementation of **DBNet** for scene text detection.
- **/crnn**: Implementation of **CRNN** with a CTC loss for scene text recognition.

---

## ⚙️ Setup
First, create a virtual environment and install the required dependencies.

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# Install all necessary packages
pip install -r requirements.txt
🚀 Usage
## DBNet (Detection)
# To train the DBNet model (update paths in the script)
python dbnet/train.py

# To run inference on a single image
python dbnet/predict.py --model_path /path/to/your/dbnet_model.pth --image_path /path/to/your/image.jpg

##Of course. Here is the complete, well-commented code content for all the files in the Paper Implementation with Pytorch directory.

Top-Level Files
README.md
Markdown

# PyTorch Implementations of Foundational OCR Architectures

## 🎯 Objective
This directory contains from-scratch PyTorch implementations of the core algorithms used in modern OCR systems like PaddleOCR. The goal is to provide a clear, educational resource for understanding these models, focusing on architectural correctness and readability over state-of-the-art performance.

---

## 📂 Project Structure
- **/dbnet**: Implementation of **DBNet** for scene text detection.
- **/crnn**: Implementation of **CRNN** with a CTC loss for scene text recognition.

---

## ⚙️ Setup
First, create a virtual environment and install the required dependencies.

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# Install all necessary packages
pip install -r requirements.txt
🚀 Usage
DBNet (Detection)
Bash

# To train the DBNet model (update paths in the script)
python dbnet/train.py

# To run inference on a single image
python dbnet/predict.py --model_path /path/to/your/dbnet_model.pth --image_path /path/to/your/image.jpg
##CRNN (Recognition)
# To train the CRNN model (update paths in the script)
python crnn/train.py

# To run inference on a single cropped word image
python crnn/predict.py --model_path /path/to/your/crnn_model.pth --image_path /path/to/your/word.png
