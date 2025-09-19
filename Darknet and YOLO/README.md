# Training YOLOv3 with the Darknet Framework

## 🎯 Objective
This directory provides a hands-on guide to training a custom **YOLOv3 (You Only Look Once)** object detector using the **Darknet** neural network framework. Darknet is a fast, lightweight framework written in C and CUDA, making it highly efficient for training and deployment.

We will cover the complete workflow: setting up the framework, preparing the data, configuring the model, and running training and inference.

---

## ⚙️ 1. Framework Setup
The first step is to clone and build the original Darknet repository.

```bash
# Clone the official Darknet repository
git clone [https://github.com/pjreddie/darknet](https://github.com/pjreddie/darknet)
cd darknet

# Compile the source code
# Set GPU=1 in the Makefile if you have an Nvidia GPU and CUDA installed
# For GPU support: nano Makefile, set GPU=1, CUDNN=1, OPENCV=1, then save
make
