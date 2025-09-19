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
