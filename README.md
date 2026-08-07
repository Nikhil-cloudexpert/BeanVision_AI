☕ BeanVisionAI

<div align="center">

BeanVisionAI
AI-Powered Coffee Bean Quality Assessment System










An intelligent computer vision pipeline for automated coffee bean segmentation, feature extraction, classification, and quality assessment.

</div>

📖 Overview

BeanVisionAI is a deep learning-based coffee bean quality assessment system designed to automate the visual inspection and classification of coffee beans.

Manual coffee bean inspection can be time-consuming and dependent on human expertise. BeanVisionAI addresses this challenge by combining instance segmentation, self-supervised visual feature extraction, and neural-network-based classification into a modular AI pipeline.

Instead of relying on a single model for the entire task, BeanVisionAI separates the problem into specialized stages:

YOLOv8 identifies and segments individual coffee beans.
The segmentation masks are used to isolate individual beans from their background.
DINOv2 extracts high-level visual representations from the isolated bean images.
A custom Multi-Layer Perceptron (MLP) processes the extracted feature embeddings.
The MLP predicts one of the 7 defined coffee bean classes.
The resulting predictions can be used for automated quality analysis and reporting.

The project is designed as a research-oriented computer vision system with a modular architecture that can later be extended toward real-time inspection, edge AI, APIs, dashboards, and industrial automation.

🎯 Project Objectives

BeanVisionAI aims to:

Automate coffee bean quality inspection.
Segment individual coffee beans from images.
Extract meaningful visual representations from segmented beans.
Classify beans into predefined quality and defect categories.
Reduce dependency on manual visual inspection.
Provide a reproducible deep learning pipeline.
Enable future deployment on resource-constrained edge devices.
Create a foundation for intelligent coffee processing and sorting systems.
✨ Features
🔍 Computer Vision
Coffee bean detection
Instance segmentation
Individual bean extraction
Mask-based background removal
Image preprocessing
Prediction visualization
🧠 Deep Learning
YOLOv8 instance segmentation
DINOv2 visual feature extraction
768-dimensional feature embeddings
Custom MLP classification
Transfer learning / pretrained visual representations
📊 Classification

BeanVisionAI currently supports 7 classes:

cut
good
husk
immature
parchment
partial-black
shell
⚙️ Engineering
Modular training pipeline
Separate feature extraction and classification stages
Dataset preprocessing
Model evaluation
Model export using ONNX
Reproducible experimentation
🚀 Future Deployment
REST API
Web dashboard
Docker containerization
Real-time camera inference
Edge AI deployment
Industrial coffee sorting integration
🧠 AI Architecture

BeanVisionAI uses a multi-stage AI architecture instead of relying on a single classification model.

                    ☕ Coffee Bean Image
                            │
                            ▼
                  Image Preprocessing
                            │
                            ▼
               ┌────────────────────────┐
               │ YOLOv8 Segmentation    │
               │                        │
               │ Detection + Masks      │
               └────────────┬───────────┘
                            │
                            ▼
                  Individual Bean Masks
                            │
                            ▼
                   Bean Extraction
                            │
                            ▼
                  Background Removal
                            │
                            ▼
               ┌────────────────────────┐
               │        DINOv2          │
               │                        │
               │ Visual Feature         │
               │ Extraction             │
               └────────────┬───────────┘
                            │
                            ▼
                   768-D Feature Vector
                            │
                            ▼
               ┌────────────────────────┐
               │    MLP Classifier      │
               │                        │
               │ 768 → 512 → 256 → 7   │
               └────────────┬───────────┘
                            │
                            ▼
                    7-Class Prediction
                            │
                            ▼
                  Quality Assessment
                            │
                            ▼
                    Final AI Report
🔬 Model Components
1. YOLOv8 Instance Segmentation

YOLOv8 is used as the segmentation stage of BeanVisionAI.

Its primary responsibility is to identify individual coffee beans and generate segmentation masks around them.

Responsibilities
Detect individual beans
Locate beans within an image
Generate instance masks
Separate overlapping beans
Provide bean-level regions for downstream processing

The resulting masks are then used to isolate individual beans before feature extraction.

2. Individual Bean Extraction

After YOLOv8 generates the segmentation masks, BeanVisionAI extracts individual bean regions from the original image.

The extraction stage allows the subsequent feature extractor to focus on the bean itself rather than unnecessary background information.

Original Image
      │
      ▼
YOLOv8 Mask
      │
      ▼
Mask Application
      │
      ▼
Background Removal
      │
      ▼
Individual Bean Image

This creates a cleaner input for DINOv2.

🧬 DINOv2 Feature Extraction

DINOv2 is used as the visual feature extraction component of BeanVisionAI.

Rather than directly training a classifier on raw images, the system uses DINOv2 to transform each isolated bean image into a high-dimensional visual representation.

The extracted representation captures visual characteristics that can be useful for distinguishing different coffee bean categories.

Feature Extraction Pipeline
Individual Bean Image
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
Stored Feature Dataset

These extracted features are subsequently used to train the MLP classifier.

Why DINOv2?

Using a pretrained visual representation model provides a feature extraction stage that can capture complex visual information such as:

Shape
Texture
Surface appearance
Color patterns
Structural characteristics
Visual differences between bean categories

DINOv2 therefore acts as the feature extractor, not the final classifier.

🧠 MLP Classification

The extracted DINOv2 feature vectors are passed to a custom Multi-Layer Perceptron (MLP) classifier.

The current conceptual architecture is:

DINOv2 Feature Vector
        │
        ▼
      768
        │
        ▼
      512
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
      256
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
       7 Classes
Classification Output

The MLP produces predictions for exactly 7 classes.

0 → cut
1 → good
2 → husk
3 → immature
4 → parchment
5 → partial-black
6 → shell
🏷️ Dataset Classes

BeanVisionAI uses 7 coffee bean classes.

Class ID	Class Name
0	cut
1	good
2	husk
3	immature
4	parchment
5	partial-black
6	shell
Class Mapping
CLASS_NAMES = {
    0: "cut",
    1: "good",
    2: "husk",
    3: "immature",
    4: "parchment",
    5: "partial-black",
    6: "shell"
}

Important: These are the official seven classes used by the BeanVisionAI classification pipeline. No additional classes such as broken, black, or other are part of the current 7-class classification setup.

📂 Dataset Organization

The dataset is organized into separate subsets for model development and evaluation.

Dataset/
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

The project can use annotated images for the YOLOv8 segmentation stage and subsequently generate isolated bean samples for DINOv2 feature extraction.

🏗️ Project Structure
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
│   ├── 02_yolov8_training.ipynb
│   ├── 03_dinov2_features.ipynb
│   ├── 04_mlp_training.ipynb
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
🚀 Tech Stack
Category	Technology	Purpose
Programming	Python	Core development
Deep Learning	PyTorch	Model development and training
Segmentation	YOLOv8	Coffee bean instance segmentation
Feature Extraction	DINOv2	Visual feature representation
Classification	MLP	7-class bean classification
Computer Vision	OpenCV	Image processing and visualization
Numerical Computing	NumPy	Feature and array operations
Data Analysis	Pandas	Dataset and experiment analysis
Visualization	Matplotlib	Graphs and result visualization
ML Evaluation	scikit-learn	Classification metrics and evaluation
Annotation	Roboflow	Dataset annotation and preparation
Experimentation	Jupyter Notebook	Research and experimentation
GPU Training	Google Colab	Model training and experimentation
Model Export	ONNX	Portable model deployment
Version Control	Git	Source control
Repository	GitHub	Collaboration and project management
⚙️ Installation
1. Clone the Repository
git clone https://github.com/yourusername/BeanVisionAI.git

Move into the project directory:

cd BeanVisionAI
2. Create a Virtual Environment
Windows
python -m venv venv
venv\Scripts\activate
Linux / macOS
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt

If you're using a CUDA-enabled environment, install the appropriate PyTorch version for your GPU environment.

🧪 Development Environment

BeanVisionAI can be developed and trained using:

Local Python environments
Jupyter Notebook
Google Colab
CUDA-enabled GPUs

For large-scale model training and feature extraction, GPU acceleration is recommended.

🏋️ YOLOv8 Training

The first stage of the pipeline is training the YOLOv8 segmentation model.

Example:

yolo task=segment mode=train \
model=yolov8n-seg.pt \
data=data.yaml \
epochs=100 \
imgsz=640 \
batch=16

The trained model can then be used to generate segmentation masks for individual coffee beans.

🧬 DINOv2 Feature Extraction

Once segmentation is complete, individual bean images can be passed to DINOv2.

The feature extraction process is:

Segmented Bean
      │
      ▼
Preprocessing
      │
      ▼
DINOv2
      │
      ▼
Feature Embedding
      │
      ▼
768-D Vector
      │
      ▼
Saved Feature Dataset

Features can be stored for subsequent MLP training.

Example output structure:

DINO_Features/
│
├── train/
├── valid/
└── test/
🧠 MLP Training

The extracted DINOv2 embeddings are used as input to the MLP classifier.

Input
768 Features
     │
     ▼
Dense Layer
512
     │
     ▼
BatchNorm + ReLU + Dropout
     │
     ▼
Dense Layer
256
     │
     ▼
BatchNorm + ReLU + Dropout
     │
     ▼
Output Layer
7 Classes

The classifier predicts:

cut
good
husk
immature
parchment
partial-black
shell
🔍 Inference

BeanVisionAI supports image-based inference.

Example:

python inference.py --source image.jpg

For a complete folder:

python inference.py --source test_images/

The inference pipeline performs:

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
MLP Classification
     │
     ▼
Class Prediction
📊 Evaluation

BeanVisionAI evaluates the segmentation and classification stages independently.

Segmentation Metrics

The YOLOv8 segmentation stage can be evaluated using:

Precision
Recall
mAP@50
mAP@50–95
IoU
Classification Metrics

The MLP classifier can be evaluated using:

Accuracy
Precision
Recall
F1 Score
Confusion Matrix
Per-class performance
📈 Model Performance

Performance values should be reported using results from the final validated experiments.

Model Component	Metric	Value
YOLOv8 Segmentation	Precision	TBD
YOLOv8 Segmentation	Recall	TBD
YOLOv8 Segmentation	mAP@50	TBD
YOLOv8 Segmentation	mAP@50–95	TBD
MLP Classification	Accuracy	TBD
MLP Classification	Macro F1	TBD

Note: Benchmark values should only be added after the corresponding experiment has been finalized and reproduced.

🖼️ Example Prediction

A typical prediction can be represented as:

Input Image
      │
      ▼
YOLOv8 detects individual beans
      │
      ▼
Bean 1 → Segmented
Bean 2 → Segmented
Bean 3 → Segmented
Bean 4 → Segmented
      │
      ▼
DINOv2 extracts features
      │
      ▼
MLP classification
      │
      ▼

Bean 1 → good
Bean 2 → cut
Bean 3 → husk
Bean 4 → partial-black
📷 Visual Results

Add project screenshots and prediction examples here.

Recommended structure:

assets/
└── images/
    ├── input_example.png
    ├── segmentation_result.png
    ├── extracted_beans.png
    ├── classification_result.png
    └── confusion_matrix.png

Example:

![YOLOv8 Segmentation](assets/images/segmentation_result.png)
🔄 Complete Workflow
┌───────────────────────┐
│   Coffee Bean Image   │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│   Preprocessing       │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ YOLOv8 Segmentation   │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ Individual Bean       │
│ Extraction            │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ Background Removal    │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ DINOv2 Feature        │
│ Extraction            │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ 768-D Feature Vector  │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ MLP Classifier        │
│ 768 → 512 → 256 → 7  │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ Bean Class Prediction │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ Quality Assessment    │
└───────────────────────┘
📦 Model Export

BeanVisionAI supports exporting trained models to portable formats such as ONNX.

ONNX export can help facilitate:

Cross-platform inference
Edge deployment
Runtime optimization
Integration with different inference environments

Example:

model.export(format="onnx")
🚀 Deployment Roadmap

The architecture is designed to support future deployment beyond notebooks.

Research Prototype
        │
        ▼
Local Inference
        │
        ▼
ONNX Export
        │
        ▼
API Layer
        │
        ▼
Web Dashboard
        │
        ▼
Docker Deployment
        │
        ▼
Edge AI
        │
        ▼
Industrial Inspection
🎯 Roadmap
Phase 1 — Dataset & Infrastructure

Repository initialization

Project structure

Dataset preparation

Seven-class definition

Dataset expansion

Dataset quality analysis

Phase 2 — Segmentation

YOLOv8 segmentation pipeline

Bean mask generation

Individual bean extraction

Hyperparameter optimization

Segmentation benchmarking

Phase 3 — Feature Extraction

DINOv2 integration

Bean feature extraction

768-dimensional feature generation

Feature analysis

Feature visualization

Phase 4 — Classification

MLP classifier architecture

7-class classification

Hyperparameter optimization

Cross-validation

Detailed error analysis

Phase 5 — Deployment

ONNX export exploration

Optimized inference

FastAPI backend

Streamlit dashboard

Docker containerization

Phase 6 — Edge & Industrial AI

Edge inference

Real-time camera pipeline

Jetson deployment

Raspberry Pi deployment

Conveyor belt inspection

Automated bean sorting

🌱 Future Scope

BeanVisionAI can be expanded into a complete intelligent coffee inspection platform.

Real-Time Inspection

Integrate cameras to classify coffee beans continuously in real time.

Industrial Sorting

Connect the AI system with mechanical sorting mechanisms to automatically separate defective beans.

Edge AI

Deploy optimized models on:

NVIDIA Jetson
Raspberry Pi
Industrial edge computers
Other supported embedded platforms
Cloud Integration

Future versions can provide:

Centralized inference
Quality monitoring
Historical analytics
Dataset management
Model management
Intelligent Reporting

Generate automated reports containing:

Total beans analyzed
Class distribution
Defect percentage
Quality statistics
Model confidence
Batch-level analysis
🧪 Research Opportunities

BeanVisionAI can serve as a foundation for further research into:

Agricultural computer vision
Coffee bean defect classification
Vision Transformers
Self-supervised learning
Instance segmentation
Feature representation learning
Edge AI
Automated agricultural inspection
Intelligent sorting systems
🤝 Contributing

Contributions are welcome.

To contribute:

1. Fork the repository
git fork https://github.com/yourusername/BeanVisionAI
2. Clone your fork
git clone https://github.com/yourusername/BeanVisionAI.git
3. Create a branch
git checkout -b feature/your-feature
4. Make your changes

Implement and test your contribution.

5. Commit your changes
git add .
git commit -m "Add: your feature"
6. Push your branch
git push origin feature/your-feature
7. Open a Pull Request

Please provide:

A clear description
Motivation for the change
Testing information
Screenshots where appropriate
🐛 Issues & Feature Requests

If you discover a bug or want to propose a feature, please use the GitHub issue templates.

When reporting an issue, include:

Operating system
Python version
GPU/CPU information
Relevant model version
Steps to reproduce
Error logs
Screenshots when applicable
📜 License

BeanVisionAI is released under the MIT License.

See the LICENSE file for the complete license terms.

🙏 Acknowledgements

BeanVisionAI builds upon several open-source technologies and research directions, including:

YOLOv8
DINOv2
PyTorch
OpenCV
NumPy
scikit-learn
Roboflow
ONNX

The project would not be possible without the broader open-source computer vision and machine learning community.

📚 Citation

If BeanVisionAI is used in academic research, projects, or publications, please cite the repository.

@software{beanvisionai,
  title  = {BeanVisionAI: AI-Powered Coffee Bean Quality Assessment System},
  author = {Nikhil},
  year   = {2026},
  url    = {https://github.com/yourusername/BeanVisionAI}
}

Replace the repository URL with the actual GitHub repository URL before publishing.

👨‍💻 Author

Nikhil

B.Tech Computer Science & Engineering
Artificial Intelligence & Machine Learning

GLA University, Mathura

📬 Contact

For questions, suggestions, research collaboration, or technical discussions, please open a GitHub issue or discussion in the repository.

⭐ Support the Project

If you find BeanVisionAI useful:

⭐ Star the repository

🍴 Fork the project

🐛 Report bugs

💡 Suggest features

🤝 Contribute

📢 Share the project

<div align="center">

☕ BeanVisionAI
Making Coffee Bean Quality Assessment Smarter with AI

YOLOv8 × DINOv2 × MLP

Built with ❤️ using Python, PyTorch, Computer Vision, and Open Source.

</div>
.
