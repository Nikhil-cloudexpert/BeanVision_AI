# ☕ BeanVisionAI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Segmentation-red?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange?style=for-the-badge&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

### **AI-Powered Coffee Bean Quality Assessment System**

*Intelligent coffee bean detection, segmentation, classification, and quality analysis using Computer Vision and Deep Learning.*

</div>

---

# 📖 Overview

**BeanVisionAI** is an open-source AI project designed to automate coffee bean quality inspection using state-of-the-art computer vision models.

The project aims to replace manual inspection with a fast, accurate, and scalable AI pipeline capable of:

- Detecting coffee beans
- Segmenting individual beans
- Classifying bean quality
- Identifying defects
- Producing quality reports
- Enabling future industrial deployment

This project is being developed as a research-oriented system with production-ready architecture.

---

# ✨ Features

- ✅ Coffee Bean Detection
- ✅ Instance Segmentation (YOLOv8)
- ✅ Defect Identification
- ✅ Bean Classification
- ✅ Dataset Management
- ✅ Training Pipeline
- ✅ Model Evaluation
- ✅ Inference on Images
- ✅ Modular Project Structure
- ✅ Future API Support
- ✅ Future Web Dashboard
- ✅ Edge AI Deployment (Planned)

---

# 🏗 Project Structure

```text
BeanVisionAI/
│
├── .github/
│   ├── ISSUE_TEMPLATE/
│   ├── workflows/
│   └── PULL_REQUEST_TEMPLATE.md
│
├── assets/
│
├── configs/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── datasets/
│
├── docs/
│
├── models/
│   ├── checkpoints/
│   ├── exported/
│   └── weights/
│
├── notebooks/
│
├── outputs/
│   ├── predictions/
│   ├── reports/
│   └── visualizations/
│
├── src/
│   ├── data/
│   ├── inference/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── utils/
│   └── visualization/
│
├── tests/
│
├── requirements.txt
├── LICENSE
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
└── README.md
```

---

# 🚀 Tech Stack

| Category | Technologies |
|----------|--------------|
| Language | Python |
| Deep Learning | PyTorch |
| Computer Vision | OpenCV |
| Detection | YOLOv8 |
| Image Processing | NumPy |
| Visualization | Matplotlib |
| Annotation | Roboflow |
| Notebook | Jupyter |
| Version Control | Git & GitHub |

---

# ⚙ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/BeanVisionAI.git
```

Move into the project:

```bash
cd BeanVisionAI
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it.

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 📂 Dataset

The dataset contains annotated coffee bean images for:

- Good Beans
- Broken Beans
- Black Beans
- Husk
- Parchment
- Immature Beans
- Shell
- Other Defects

Supported annotation formats:

- YOLOv8
- COCO
- Pascal VOC

---

# 🧠 Model Pipeline

```
Coffee Bean Images
          │
          ▼
 Image Preprocessing
          │
          ▼
   YOLOv8 Segmentation
          │
          ▼
 Individual Bean Extraction
          │
          ▼
 Feature Analysis
          │
          ▼
 Bean Classification
          │
          ▼
 Quality Assessment
          │
          ▼
 Final Report
```

---

# 📊 Training

Example:

```bash
python train.py
```

or

```bash
yolo task=segment mode=train \
model=yolov8n-seg.pt \
data=data.yaml \
epochs=100 \
imgsz=640
```

---

# 🔍 Inference

Example:

```bash
python inference.py --source image.jpg
```

or

```bash
python inference.py --source folder/
```

---

# 📈 Evaluation

Metrics include:

- Precision
- Recall
- mAP@50
- mAP@50-95
- IoU
- Confusion Matrix
- Precision-Recall Curve

---

# 📷 Example Output

```
Input Image
      │
      ▼
Bean Detection

✔ Bean 1 → Good
✔ Bean 2 → Broken
✔ Bean 3 → Husk
✔ Bean 4 → Black

Overall Quality Score:
92%
```

---

# 🎯 Roadmap

## Phase 1

- [x] Repository Setup
- [x] Project Structure
- [ ] Dataset Collection
- [ ] Annotation

## Phase 2

- [ ] YOLOv8 Training
- [ ] Model Evaluation
- [ ] Performance Optimization

## Phase 3

- [ ] REST API
- [ ] Web Dashboard
- [ ] Mobile Support
- [ ] Docker Deployment

## Phase 4

- [ ] Edge AI
- [ ] Raspberry Pi Support
- [ ] Jetson Deployment
- [ ] Cloud Integration

---

# 🤝 Contributing

Contributions are welcome!

If you'd like to contribute:

1. Fork the repository.
2. Create a new feature branch.
3. Commit your changes.
4. Push to your fork.
5. Open a Pull Request.

Please read `CONTRIBUTING.md` before contributing.

---

# 📜 License

This project is licensed under the MIT License.

See the `LICENSE` file for more details.

---

# 💡 Future Scope

- Real-time coffee bean grading
- Industrial conveyor belt inspection
- Robotic sorting
- Edge AI deployment
- Cloud dashboard
- Automated quality reports
- Mobile application
- Multi-camera inspection
- AI-assisted defect explanation

---

# 👨‍💻 Author

**Nikhil**

B.Tech CSE (AI & ML)

GLA University, Mathura

---

# ⭐ Support

If you found this project helpful:

⭐ Star the repository

🍴 Fork the project

🐞 Report bugs

💡 Suggest new features

📢 Share the project

---

<div align="center">

### ☕ BeanVisionAI

**Making Coffee Quality Assessment Smarter with AI**

Made with ❤️ using Python, YOLOv8, and Open Source.

</div>