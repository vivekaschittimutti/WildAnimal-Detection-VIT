Wild Animal Detection Using Vision Transformers

Overview

Wild animal detection plays a crucial role in preventing human-wildlife conflicts and ensuring the safety of both animals and humans. This project aims to develop an advanced deep learning model using Vision Transformers (ViTs) to detect and classify wild animals from real-time camera feeds. The model is designed to enhance wildlife monitoring and alert systems.

Project Goals

.Enhance global feature learning to improve adaptability across diverse environments and animal species.

.Utilize larger datasets to improve generalization and detection performance.

.Ensure robustness for different environments, including varying lighting, backgrounds, and weather conditions.

Team Members

C. Vivekanandha Reddy

K. Vishnuvardhan Reddy

D. Sandeep Kumar

Project Supervisors

Dr. P. Penchala Prasad (Associate Professor, CSE-DS)

Mr. Y. P. Srinath Reddy (Assistant Professor, CSE-DS)

Methodology

1. Dataset Collection & Preprocessing

Source: Wild animal image dataset from Kaggle and other repositories.

File Formats: Images in JPEG/PNG format.

Preprocessing Steps:

Image resizing and normalization.

Data augmentation (rotation, flipping, brightness adjustments).

Splitting into training, validation, and testing sets.

2. Model Development

Architecture: Vision Transformer (ViT) for high-resolution image processing.

Feature Extraction: Self-attention mechanism for improved feature learning.

Training Setup:

Optimizer: AdamW

Loss Function: Cross-entropy loss

Batch Size: 32

Epochs: 50

3. Model Deployment

Platform: FastAPI for real-time inference.

Deployment:

Model saved in ONNX/TorchScript format.

API integration for alert systems.

Optimized inference pipeline for edge devices.

4. Evaluation Metrics

Accuracy: Measures overall model performance.

Precision & Recall: Ensures minimal false positives and false negatives.

F1-Score: Balances precision and recall.

Confusion Matrix: Evaluates model classification errors.

Results

Training Accuracy: 92.3%

Validation Accuracy: 84.7%

Real-time Inference Speed: 0.15s per image

Conclusion

This project demonstrates the effectiveness of Vision Transformers for wild animal detection. The model provides accurate and efficient classification, making it a valuable tool for wildlife monitoring. Future improvements will focus on further dataset expansion, fine-tuning, and edge-device optimization for real-world applications.
