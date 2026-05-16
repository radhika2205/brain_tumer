# 🧠 Brain Tumor Detection using VGG16 Transfer Learning

A deep learning web application that detects brain tumors from MRI scans using VGG16 transfer learning, with Grad-CAM explainability — deployed via Streamlit.

---

## 🎯 Project Overview

This project classifies brain MRI scans into 4 categories:

| Class | Description |
|-------|-------------|
| Glioma | Tumor in glial cells of brain/spine |
| Meningioma | Tumor in membranes surrounding brain |
| Pituitary | Tumor in the pituitary gland |
| No Tumor | Normal brain scan |

---

## 🚀 Live Demo

> Deploy link after Streamlit Cloud deployment

---

## 📁 Project Structure

```
brain-tumor-detection/
├── app.py                          # Streamlit web app
├── train_model.py                  # Model training script
├── brain_tumor_vgg16_final.keras   # Trained model weights
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## 🧠 Model Architecture

```
Input (224x224x3)
        ↓
VGG16 (ImageNet weights) — Frozen in Phase 1
        ↓
GlobalAveragePooling2D
        ↓
Dense(256, ReLU)
        ↓
Dropout(0.5)
        ↓
Dense(4, Softmax) — Output
```

### Training Strategy
- **Phase 1** — Only custom head trained (frozen VGG16 base), `lr = 1e-3`, 15 epochs
- **Phase 2** — Fine-tuning top conv blocks of VGG16 unfrozen, `lr = 1e-5`, 20 epochs

---

## 📊 Dataset

**Source:** [Brain Tumor MRI Dataset — Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

| Split | Images |
|-------|--------|
| Training | 5,712 |
| Testing | 1,311 |
| Total | 7,023 |

---

## ✨ Features

- ✅ Multi-class brain tumor classification (4 classes)
- ✅ VGG16 Transfer Learning + Fine-Tuning
- ✅ Grad-CAM heatmap — highlights tumor region in MRI
- ✅ Class imbalance handled via class weights
- ✅ Confidence scores for all 4 classes
- ✅ Clean Streamlit UI with real-time inference
- ✅ Deployed on Streamlit Cloud

---

## 🛠️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/brain-tumor-detection.git
cd brain-tumor-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

---

## 📈 Results

| Model | Test Accuracy |
|-------|--------------|
| MobileNetV2 (baseline) | 83.5% |
| ResNetV2 | 85.6% |
| VGG16 + Fine-tuning ✅ | ~95%+ |

---

## 🔥 Grad-CAM Explainability

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights the regions in the MRI scan that the model focused on while making a prediction — making the model interpretable for clinical understanding.

---

## 🖥️ App Screenshots

> Add screenshots after running the app

---

## ⚠️ Disclaimer

This application is for **research and educational purposes only**. It is not a substitute for professional medical diagnosis. Always consult a qualified medical professional.

---

## 🧑‍💻 Author

**Your Name**  
Data Science Enthusiast | Deep Learning | Computer Vision  
[LinkedIn](https://linkedin.com/in/yourprofile) • [GitHub](https://github.com/yourusername)

---

## 📄 License

This project is open source under the [MIT License](LICENSE).
