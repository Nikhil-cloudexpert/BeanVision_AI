# ☕ BeanVisionAI

<div align="center">

# **BeanVisionAI**

### **AI-Powered Coffee Bean Quality Assessment System**

**YOLOv8 Segmentation • DINOv2 Feature Extraction • MLP Classification**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge\&logo=pytorch\&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Segmentation-7B68EE?style=for-the-badge)
![DINOv2](https://img.shields.io/badge/DINOv2-Feature%20Extraction-FF6F00?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge\&logo=opencv\&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-Deployment-005CED?style=for-the-badge\&logo=onnx\&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-Version%20Control-181717?style=for-the-badge\&logo=github)
![License](https://img.shields.io/badge/License-MIT-F5C518?style=for-the-badge)

**An end-to-end computer vision and deep learning pipeline for detecting, segmenting, extracting, and classifying coffee beans.**

<br>

[🚀 Features](#-features) •
[🧠 Architecture](#-ai-architecture) •
[📊 Results](#-model-performance) •
[⚙️ Installation](#️-installation) •
[🛣️ Roadmap](#️-roadmap)

</div>

---

# 📖 Overview

**BeanVisionAI** is a research-oriented artificial intelligence system for automated **coffee bean quality assessment** using computer vision and deep learning.

Coffee bean quality inspection is traditionally performed through manual visual examination. This process can be time-consuming, subjective, difficult to scale, and dependent on human expertise.

BeanVisionAI approaches the problem as a **multi-stage computer vision pipeline** rather than relying on a single model.

The system combines:

* **YOLOv8 Instance Segmentation** for identifying and segmenting individual coffee beans.
* **Mask-based bean extraction** for isolating each bean from the original image.
* **DINOv2** for extracting rich visual representations from individual bean images.
* **768-dimensional feature embeddings** generated from DINOv2.
* A custom **Multi-Layer Perceptron (MLP)** for final 7-class classification.
* Evaluation and visualization tools for analyzing model performance.
* **ONNX export** for future cross-platform and edge deployment.

The overall objective is to create a modular foundation that can eventually evolve into a **real-time intelligent coffee inspection and sorting system**.

---

# 🎯 Project Objectives

BeanVisionAI is designed around the following objectives:

* 🔍 Automatically identify individual coffee beans.
* 🎭 Generate accurate instance segmentation masks.
* 🫘 Extract individual beans from complex images.
* 🧬 Generate meaningful visual embeddings using DINOv2.
* 🧠 Classify coffee beans into seven predefined categories.
* 📊 Evaluate segmentation and classification performance independently.
* ⚡ Build a pipeline suitable for future real-time inference.
* 🚀 Support portable model deployment using ONNX.
* 🏭 Provide a foundation for future industrial coffee inspection and automated sorting.

---

# ✨ Features

## 🔍 Computer Vision

* Coffee bean detection
* Instance segmentation
* Individual bean mask generation
* Bean extraction
* Background removal
* Image preprocessing
* Prediction visualization

## 🧠 Deep Learning

* YOLOv8 segmentation
* DINOv2 visual feature extraction
* 768-dimensional feature embeddings
* Custom MLP classifier
* Transfer learning through pretrained visual representations

## 🏷️ Classification

BeanVisionAI currently supports **exactly 7 classes**:

|  ID | Class           |
| --: | --------------- |
| `0` | `cut`           |
| `1` | `good`          |
| `2` | `husk`          |
| `3` | `immature`      |
| `4` | `parchment`     |
| `5` | `partial-black` |
| `6` | `shell`         |

## ⚙️ Engineering

* Modular architecture
* Separate segmentation and classification stages
* Feature caching
* Training and validation pipelines
* Evaluation tools
* Visualization utilities
* ONNX model export
* Git-based version control

## 🚀 Future Deployment

* Real-time camera inference
* FastAPI backend
* Streamlit dashboard
* Docker deployment
* Cloud integration
* Edge AI
* Industrial conveyor inspection
* Automated sorting

---

# 🧠 AI Architecture

BeanVisionAI follows a **segmentation → feature extraction → classification** architecture.

```text
                         ☕ COFFEE BEAN IMAGE
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │ Image Preprocessing │
                       └──────────┬──────────┘
                                  │
                                  ▼
                 ┌──────────────────────────────┐
                 │     YOLOv8 Segmentation      │
                 │                              │
                 │ Detection + Instance Masks  │
                 └──────────────┬───────────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Individual Bean     │
                     │ Extraction          │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Background Removal  │
                     │ / Mask Processing   │
                     └──────────┬──────────┘
                                │
                                ▼
                 ┌──────────────────────────────┐
                 │            DINOv2            │
                 │                              │
                 │  Visual Feature Extraction  │
                 └──────────────┬───────────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ 768-D Feature      │
                     │ Embedding           │
                     └──────────┬──────────┘
                                │
                                ▼
                 ┌──────────────────────────────┐
                 │       MLP Classifier         │
                 │                              │
                 │       768 → 512 → 256 → 7  │
                 └──────────────┬───────────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ 7-Class Prediction  │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Quality Assessment  │
                     └──────────┬──────────┘
                                │
                                ▼
                         📊 FINAL REPORT
```

---

# 🔬 Core AI Components

## 1. YOLOv8 Instance Segmentation

YOLOv8 forms the **first major stage** of the BeanVisionAI pipeline.

Instead of directly sending the entire coffee bean image to a classifier, YOLOv8 identifies individual bean instances and generates segmentation masks.

### YOLOv8 responsibilities

* Detect individual beans
* Localize each bean
* Generate instance masks
* Separate individual beans
* Provide regions for downstream processing

### Output

For each detected bean, the segmentation model can provide:

```text
Bean Instance
├── Bounding Box
├── Segmentation Mask
├── Class Information
└── Confidence
```

These masks are subsequently used to isolate the beans before DINOv2 feature extraction.

---

# 🫘 2. Individual Bean Extraction

After segmentation, BeanVisionAI extracts each individual bean from the original image.

```text
Original Image
      │
      ▼
YOLOv8 Segmentation Mask
      │
      ▼
Mask Application
      │
      ▼
Background Removal
      │
      ▼
Individual Bean Image
```

This stage is important because the feature extractor should focus primarily on the visual characteristics of the coffee bean rather than irrelevant background information.

The resulting images are used as input to DINOv2.

---

# 🧬 3. DINOv2 Feature Extraction

**DINOv2** is used as the feature extraction stage of BeanVisionAI.

Rather than training a classifier directly from raw pixel data, BeanVisionAI uses DINOv2 to transform each isolated bean into a high-dimensional visual representation.

The resulting embedding contains learned visual information that can be used by the downstream classifier.

### Feature pipeline

```text
Individual Bean
      │
      ▼
Image Preprocessing
      │
      ▼
DINOv2
      │
      ▼
Visual Representation
      │
      ▼
768-Dimensional Feature Vector
      │
      ▼
Feature Dataset
```

### Why DINOv2?

DINOv2 can provide rich visual representations useful for capturing characteristics such as:

* Shape
* Texture
* Surface patterns
* Color distribution
* Structural appearance
* Visual differences between classes

In BeanVisionAI, **DINOv2 is the feature extractor — not the final classifier.**

---

# 🧠 4. MLP Classification

The feature vectors generated by DINOv2 are passed to a custom **Multi-Layer Perceptron (MLP)**.

The current architecture is based around:

```text
768 → 512 → 256 → 7
```

with regularization and nonlinear activation layers.

```text
             DINOv2 Embedding
                    │
                    ▼
                 768-D
                    │
                    ▼
               Linear 512
                    │
                    ▼
            Batch Normalization
                    │
                    ▼
                  ReLU
                    │
                    ▼
                Dropout
                    │
                    ▼
               Linear 256
                    │
                    ▼
            Batch Normalization
                    │
                    ▼
                  ReLU
                    │
                    ▼
                Dropout
                    │
                    ▼
                Linear 7
                    │
                    ▼
             Class Prediction
```

The MLP produces a prediction across the seven BeanVisionAI classes.

---

# 🏷️ Supported Classes

BeanVisionAI currently uses **exactly seven classes**.

| Class ID | Class Name      | Category |
| -------: | --------------- | -------- |
|      `0` | `cut`           | Defect   |
|      `1` | `good`          | Quality  |
|      `2` | `husk`          | Defect   |
|      `3` | `immature`      | Defect   |
|      `4` | `parchment`     | Defect   |
|      `5` | `partial-black` | Defect   |
|      `6` | `shell`         | Defect   |

### Python Class Mapping

```python
CLASS_NAMES = {
    0: "cut",
    1: "good",
    2: "husk",
    3: "immature",
    4: "parchment",
    5: "partial-black",
    6: "shell"
}
```

> **Important:** The BeanVisionAI classification pipeline contains only these seven classes. Classes such as `broken`, `black`, `full-black`, or `other` are **not additional BeanVisionAI classes**.

---

# 📊 Technology Stack

| Layer               | Technology           | Role                                 |
| ------------------- | -------------------- | ------------------------------------ |
| Programming         | **Python**           | Core development                     |
| Deep Learning       | **PyTorch**          | Training and inference               |
| Segmentation        | **YOLOv8**           | Bean instance segmentation           |
| Feature Extraction  | **DINOv2**           | Visual feature representation        |
| Classification      | **MLP**              | 7-class classification               |
| Computer Vision     | **OpenCV**           | Image processing                     |
| Numerical Computing | **NumPy**            | Array and feature operations         |
| Data Analysis       | **Pandas**           | Dataset and experiment analysis      |
| Visualization       | **Matplotlib**       | Graphs and result visualization      |
| Evaluation          | **scikit-learn**     | Classification metrics               |
| Annotation          | **Roboflow**         | Dataset preparation                  |
| Experimentation     | **Jupyter Notebook** | Research workflow                    |
| GPU Environment     | **Google Colab**     | Model training and experimentation   |
| Model Export        | **ONNX**             | Portable inference                   |
| Version Control     | **Git**              | Source control                       |
| Repository          | **GitHub**           | Collaboration and project management |

---

# 🗂️ Project Structure

```text
BeanVisionAI/
│
├── .github/
│   ├── ISSUE_TEMPLATE/
│   ├── workflows/
│   └── PULL_REQUEST_TEMPLATE.md
│
├── assets/
│   ├── images/
│   ├── diagrams/
│   └── demos/
│
├── configs/
│   ├── data.yaml
│   └── model_configs/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── train/
│   ├── valid/
│   └── test/
│
├── docs/
│   ├── architecture/
│   ├── dataset/
│   └── experiments/
│
├── models/
│   ├── checkpoints/
│   ├── weights/
│   ├── exported/
│   └── onnx/
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_yolov8_segmentation.ipynb
│   ├── 03_dinov2_feature_extraction.ipynb
│   ├── 04_mlp_classification.ipynb
│   └── 05_deployment.ipynb
│
├── outputs/
│   ├── predictions/
│   ├── reports/
│   ├── metrics/
│   └── visualizations/
│
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── segmentation/
│   ├── feature_extraction/
│   ├── classification/
│   ├── training/
│   ├── evaluation/
│   ├── inference/
│   ├── deployment/
│   ├── visualization/
│   └── utils/
│
├── tests/
│
├── requirements.txt
├── .gitignore
├── LICENSE
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
└── README.md
```

---

# 📂 Dataset

The dataset is organized into training, validation, and testing subsets.

```text
dataset/
│
├── train/
│   ├── images/
│   └── labels/
│
├── valid/
│   ├── images/
│   └── labels/
│
└── test/
    ├── images/
    └── labels/
```

The dataset supports the seven BeanVisionAI classes:

```text
0 → cut
1 → good
2 → husk
3 → immature
4 → parchment
5 → partial-black
6 → shell
```

---

# 🔄 Data Processing Pipeline

The complete data preparation process is:

```text
Raw Coffee Bean Images
          │
          ▼
       Annotation
          │
          ▼
    Dataset Validation
          │
          ▼
      Preprocessing
          │
          ▼
YOLOv8 Segmentation Dataset
          │
          ▼
   Individual Bean Masks
          │
          ▼
    Bean Extraction
          │
          ▼
   DINOv2 Processing
          │
          ▼
  768-D Feature Dataset
          │
          ▼
    MLP Classification
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/BeanVisionAI.git
cd BeanVisionAI
```

Replace `YOUR_USERNAME` with your GitHub username.

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

For GPU training, install the PyTorch version appropriate for your CUDA environment.

---

# 🧪 Recommended Environment

BeanVisionAI can be used in:

* 🖥️ Local Python environments
* 📓 Jupyter Notebook
* ☁️ Google Colab
* 🎮 CUDA-enabled GPU environments

GPU acceleration is recommended for:

* YOLOv8 training
* DINOv2 feature extraction
* MLP experimentation

---

# 🏋️ Training Pipeline

BeanVisionAI separates the training workflow into multiple stages.

## Stage 1 — YOLOv8 Segmentation

Train the segmentation model using the annotated dataset.

Example:

```bash
yolo task=segment mode=train \
model=yolov8n-seg.pt \
data=data.yaml \
epochs=100 \
imgsz=640 \
batch=16
```

The trained model is then used to generate bean segmentation masks.

---

# 🫘 Stage 2 — Bean Extraction

The segmentation model processes images and generates individual bean masks.

The masks are used to create isolated bean images.

```text
Input Image
     │
     ▼
YOLOv8
     │
     ▼
Instance Masks
     │
     ▼
Mask Processing
     │
     ▼
Individual Bean Images
```

---

# 🧬 Stage 3 — DINOv2 Feature Extraction

The isolated bean images are processed using DINOv2.

```text
Bean Image
     │
     ▼
DINOv2
     │
     ▼
768-D Embedding
     │
     ▼
Save Feature
```

Features can be stored for:

```text
DINO_Features/
│
├── train/
├── valid/
└── test/
```

This allows the MLP training stage to operate on precomputed feature embeddings instead of repeatedly processing images through DINOv2.

---

# 🧠 Stage 4 — MLP Training

The MLP receives the DINOv2 feature vectors.

```text
Input: 768-D
      │
      ▼
512 Neurons
      │
      ▼
256 Neurons
      │
      ▼
7 Output Classes
```

The classifier learns to distinguish the seven BeanVisionAI categories.

---

# 🔍 Inference

The complete inference pipeline is:

```text
Input Image
     │
     ▼
YOLOv8 Segmentation
     │
     ▼
Bean Extraction
     │
     ▼
DINOv2 Feature Extraction
     │
     ▼
768-D Feature Vector
     │
     ▼
MLP Classifier
     │
     ▼
Class Prediction
```

Example:

```bash
python inference.py --source image.jpg
```

Folder inference:

```bash
python inference.py --source test_images/
```

---

# 📈 Model Performance

## YOLOv8 Segmentation

The current reported YOLOv8 experiment achieved the following metrics:

| Metric    |     Result |
| --------- | ---------: |
| Precision | **97.59%** |
| Recall    | **89.27%** |
| mAP@50    | **94.87%** |
| mAP@50–95 | **92.62%** |

These values represent the reported segmentation experiment and should be treated separately from the final DINOv2 + MLP classification results.

## DINOv2 + MLP

| Metric    |  Result |
| --------- | ------: |
| Accuracy  | **TBD** |
| Precision | **TBD** |
| Recall    | **TBD** |
| F1 Score  | **TBD** |
| Macro F1  | **TBD** |

> Classification benchmark values should be added after the final MLP experiment has been evaluated on the designated test set.

---

# 📊 Evaluation Metrics

## YOLOv8 Segmentation

BeanVisionAI evaluates segmentation using:

* Precision
* Recall
* mAP@50
* mAP@50–95
* IoU

## MLP Classification

The classification stage can be evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Macro F1
* Weighted F1
* Confusion Matrix
* Per-class accuracy

---

# 📉 Confusion Matrix

A confusion matrix can be used to analyze which of the seven classes are most frequently confused.

Expected labels:

```text
cut
good
husk
immature
parchment
partial-black
shell
```

Recommended visualization:

```text
                 Predicted
             ─────────────────
             cut good husk ...
Actual cut
       good
       husk
       ...
```

Add the final confusion matrix under:

```text
assets/images/confusion_matrix.png
```

---

# 🖼️ Example Results

Store project visuals inside:

```text
assets/
└── images/
    ├── original_input.png
    ├── yolov8_segmentation.png
    ├── extracted_beans.png
    ├── dinov2_features.png
    ├── classification_result.png
    └── confusion_matrix.png
```

Then display them in the README:

```markdown
![YOLOv8 Segmentation](assets/images/yolov8_segmentation.png)

![Bean Extraction](assets/images/extracted_beans.png)

![Classification Result](assets/images/classification_result.png)
```

---

# 🧪 Example Output

A complete inference session can conceptually produce:

```text
================================================
              BeanVisionAI
        Coffee Bean Quality Analysis
================================================

Input Image:
coffee_sample_01.jpg

Beans Detected:
24

------------------------------------------------
Bean Classification
------------------------------------------------

Bean 01 → good
Bean 02 → cut
Bean 03 → good
Bean 04 → husk
Bean 05 → partial-black
Bean 06 → good
...

------------------------------------------------
Quality Summary
------------------------------------------------

Total Beans       : 24
Good              : XX
Cut               : XX
Husk              : XX
Immature          : XX
Parchment         : XX
Partial-Black     : XX
Shell             : XX

================================================
```

The exact values should be generated from the actual inference output rather than hard-coded.

---

# 📦 Model Export

BeanVisionAI supports model export for future deployment.

ONNX can provide a portable representation that can be integrated into different inference environments.

Example:

```python
model.export(format="onnx")
```

Potential deployment targets include:

* Desktop inference
* Server environments
* Docker containers
* Edge devices
* Embedded AI systems

---

# 🚀 Deployment Architecture

The long-term deployment architecture is envisioned as:

```text
                    ☕ Coffee Beans
                          │
                          ▼
                   📷 Camera Input
                          │
                          ▼
                 YOLOv8 Segmentation
                          │
                          ▼
                   Bean Extraction
                          │
                          ▼
                       DINOv2
                          │
                          ▼
                     MLP Model
                          │
                          ▼
                  Quality Prediction
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
       📊 Web Dashboard          ⚙️ Industrial
              │                       │
              ▼                       ▼
          Analytics             Sorting System
```

---

# 🌐 Future Application Architecture

The system can eventually be exposed through an API:

```text
                 Client / Camera
                        │
                        ▼
                 FastAPI Backend
                        │
                        ▼
              BeanVisionAI Pipeline
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
       YOLOv8         DINOv2         MLP
          │             │             │
          └─────────────┼─────────────┘
                        │
                        ▼
                  Quality Result
                        │
                        ▼
                 Web Dashboard
```

---

# 🛣️ Roadmap

## Phase 1 — Foundation

* [x] Repository setup
* [x] Project structure
* [x] Dataset preparation
* [x] Seven-class definition
* [x] Initial segmentation pipeline

## Phase 2 — Segmentation

* [x] YOLOv8 integration
* [x] Instance segmentation
* [x] Mask extraction
* [x] Individual bean extraction
* [x] Initial model evaluation
* [ ] Further hyperparameter optimization
* [ ] Dataset expansion

## Phase 3 — Feature Extraction

* [x] DINOv2 integration
* [x] Feature extraction pipeline
* [x] 768-D feature generation
* [x] Train/validation/test feature organization
* [ ] Feature visualization
* [ ] Feature-space analysis

## Phase 4 — Classification

* [x] MLP architecture
* [x] 7-class classification pipeline
* [ ] Hyperparameter tuning
* [ ] Cross-validation
* [ ] Detailed error analysis
* [ ] Final benchmark

## Phase 5 — Deployment

* [x] ONNX export exploration
* [ ] Optimized inference
* [ ] FastAPI backend
* [ ] Streamlit dashboard
* [ ] Docker container
* [ ] REST API

## Phase 6 — Edge AI

* [ ] Edge inference optimization
* [ ] Real-time camera pipeline
* [ ] NVIDIA Jetson deployment
* [ ] Raspberry Pi deployment
* [ ] Industrial edge computer integration

## Phase 7 — Industrial Automation

* [ ] Conveyor belt integration
* [ ] Real-time bean detection
* [ ] Automated bean sorting
* [ ] Batch-level quality scoring
* [ ] Production monitoring

---

# 🌱 Future Scope

BeanVisionAI can evolve from a research prototype into an intelligent coffee inspection platform.

### 📷 Real-Time Inspection

Use cameras to continuously analyze beans on a conveyor belt.

### 🏭 Automated Sorting

Integrate the AI predictions with actuators or robotic mechanisms to separate beans based on quality.

### ⚡ Edge AI

Deploy optimized inference on:

* NVIDIA Jetson devices
* Raspberry Pi
* Embedded AI computers
* Industrial edge systems

### ☁️ Cloud Analytics

A cloud-connected version could provide:

* Batch history
* Quality trends
* Model monitoring
* Dataset management
* Remote analytics
* Production statistics

### 📊 Automated Quality Reports

Generate reports containing:

* Total beans inspected
* Class distribution
* Defect percentage
* Good bean percentage
* Batch quality score
* Model confidence
* Historical comparisons

---

# 🔬 Research Potential

BeanVisionAI can also serve as a research platform for exploring:

* Agricultural computer vision
* Coffee bean defect detection
* Instance segmentation
* Self-supervised learning
* Vision Transformers
* Feature representation learning
* Transfer learning
* Few-shot classification
* Edge AI
* Intelligent agricultural automation

Potential future research directions include comparing:

```text
DINOv2
   │
   ├── MLP
   │
   ├── SVM
   │
   ├── Random Forest
   │
   └── Other Classifiers
```

and evaluating which feature/classifier combination provides the best accuracy, robustness, and inference efficiency.

---

# 🔐 Reproducibility

For reproducible experiments, record:

* Dataset version
* Dataset split
* Class mapping
* Model checkpoint
* Training epochs
* Batch size
* Image size
* Learning rate
* Optimizer
* Random seed
* Hardware environment
* Python version
* PyTorch version

This helps ensure that experimental results can be independently reproduced.

---

# 🤝 Contributing

Contributions are welcome!

If you'd like to contribute:

### 1. Fork the repository

### 2. Clone your fork

```bash
git clone https://github.com/YOUR_USERNAME/BeanVisionAI.git
cd BeanVisionAI
```

### 3. Create a feature branch

```bash
git checkout -b feature/your-feature
```

### 4. Make your changes

Implement your feature and test it thoroughly.

### 5. Commit your changes

```bash
git add .
git commit -m "Add: your feature"
```

### 6. Push your branch

```bash
git push origin feature/your-feature
```

### 7. Open a Pull Request

Please include:

* Clear description of the change
* Motivation
* Testing information
* Screenshots where applicable
* Relevant benchmark results

For major architectural changes, open an issue before implementation so the approach can be discussed.

---

# 🐛 Issues & Bug Reports

If you find a bug, create a GitHub issue and include:

* Operating system
* Python version
* PyTorch version
* GPU/CPU information
* Dataset version
* Model checkpoint
* Error message
* Steps to reproduce
* Relevant screenshots

---

# 📜 License

BeanVisionAI is released under the **MIT License**.

The MIT License permits use, modification, distribution, and private or commercial use subject to the license conditions.

See the [`LICENSE`](LICENSE) file for the complete license text.

---

# 🙏 Acknowledgements

BeanVisionAI builds upon the work of the open-source machine learning and computer vision community.

The project uses or builds upon technologies including:

* **YOLOv8** for instance segmentation
* **DINOv2** for visual feature extraction
* **PyTorch** for deep learning
* **OpenCV** for computer vision
* **NumPy** for numerical computation
* **scikit-learn** for evaluation
* **Roboflow** for dataset preparation
* **ONNX** for portable model deployment

Special appreciation goes to the researchers and developers who make modern computer vision research accessible through open-source software.

---

# 📚 Citation

If BeanVisionAI is used in academic work, research, presentations, or derivative projects, please cite the repository.

```bibtex
@software{beanvisionai,
  title  = {BeanVisionAI: AI-Powered Coffee Bean Quality Assessment System},
  author = {Nikhil},
  year   = {2026},
  url    = {https://github.com/YOUR_USERNAME/BeanVisionAI}
}
```

Replace the repository URL with the final GitHub repository address before publishing.

---

# 👨‍💻 Author

<div align="center">

### **Nikhil**

**B.Tech Computer Science & Engineering — Artificial Intelligence & Machine Learning**

**GLA University, Mathura**

</div>

---

# ⭐ Support BeanVisionAI

If you find this project useful:

⭐ **Star the repository**

🍴 **Fork the project**

🐛 **Report bugs**

💡 **Suggest improvements**

🤝 **Contribute**

📢 **Share the project**

Every contribution helps improve the project.

---

<div align="center">

# ☕ BeanVisionAI

### **Making Coffee Bean Quality Assessment Smarter with AI**

**YOLOv8 × DINOv2 × MLP**

*Segment → Extract → Classify → Assess*

<br>

**Built with Python, PyTorch, Computer Vision, and Open Source ❤️**

</div>
