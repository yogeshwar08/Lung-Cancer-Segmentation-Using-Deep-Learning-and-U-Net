# LungVision AI

## Lung CT Image Segmentation & Classification

LungVision AI is an end-to-end deep learning and computer vision
project that processes lung CT images through preprocessing,
Modified U-Net segmentation, ROI extraction, deep learning
classification, and Flask deployment.

## Pipeline

```text
JPEG CT Image
      ↓
Image Preprocessing
      ↓
Modified U-Net Segmentation
      ↓
ROI Extraction
      ↓
Deep Learning Classification
      ↓
Normal / Benign / Malignant
      ↓
Flask Web Application