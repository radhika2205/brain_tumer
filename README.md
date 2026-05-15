# 🧠 Brain Tumor Detection — VGG16 + Grad-CAM + Streamlit

Deep learning project to detect brain tumors from MRI scans using
VGG16 CNN with Grad-CAM explainability, deployed on Streamlit.

---

## 📁 Project Structure

```
brain_tumor_detection/
│
├── dataset/
│   ├── train/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── no_tumor/
│   │   └── pituitary/
│   ├── val/
│   │   └── (same 4 folders)
│   └── test/
│       └── (same 4 folders)
│
├── model/
│   ├── cnn_model.py          ← VGG16 architecture + fine-tuning
│   ├── best_model.h5         ← saved after training
│   ├── class_indices.json    ← label mapping
│   └── training_history.png  ← accuracy/loss plots
│
├── gradcam/
│   └── gradcam.py            ← Grad-CAM heatmap (block5_conv3)
│
├── utils/
│   └── data_preprocessing.py ← augmentation + preprocessing
│
├── app/
│   └── streamlit_app.py      ← Streamlit web app
│
├── train.py                  ← main training script
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Download Dataset
Link: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Place images into:
```
dataset/train/glioma/        ← glioma training images
dataset/train/meningioma/
dataset/train/no_tumor/
dataset/train/pituitary/
dataset/val/...              ← validation images
dataset/test/...             ← test images
```

### Step 3 — Train the Model
```bash
python train.py
```

### Step 4 — Run Streamlit App
```bash
streamlit run app/streamlit_app.py
```

---

## 🏗️ VGG16 Architecture Used

```
Input (224 × 224 × 3)
        ↓
VGG16 Base (pretrained ImageNet)
  Block 1: conv1_1 → conv1_2 → MaxPool
  Block 2: conv2_1 → conv2_2 → MaxPool
  Block 3: conv3_1 → conv3_2 → conv3_3 → MaxPool
  Block 4: conv4_1 → conv4_2 → conv4_3 → MaxPool
  Block 5: conv5_1 → conv5_2 → conv5_3 → MaxPool  ← fine-tuned
        ↓
Flatten
Dense(4096, ReLU) → Dropout(0.5)
Dense(4096, ReLU) → Dropout(0.5)
Dense(1024, ReLU) → Dropout(0.3)
Dense(4, Softmax) ← output: 4 classes
```

---

## 🔥 Grad-CAM

- Layer used: `block5_conv3` (last conv layer of VGG16)
- Shows heatmap overlay on MRI highlighting the exact tumor region
- Red = high attention, Blue = low attention

---

## 📊 Training Strategy

| Phase   | What happens                    | LR     |
|---------|---------------------------------|--------|
| Phase 1 | VGG16 frozen, train head only   | 1e-4   |
| Phase 2 | Unfreeze block5, fine-tune      | 1e-5   |

---

## 📈 Expected Results

| Metric        | Value    |
|--------------|----------|
| Test Accuracy | ~95-97%  |
| Classes       | 4        |
| Dataset size  | 7,023    |
| Input size    | 224×224  |

---

## 🖥️ Streamlit App Features

- Upload MRI image (JPG/PNG)
- Instant prediction with confidence %
- Grad-CAM heatmap overlay
- All 4 class probabilities
- Tumor description + severity info

---

## ⚠️ Disclaimer
Educational and research purposes only.
Not a substitute for professional medical diagnosis.

---

## 📝 Resume Line
> "Built VGG16-based brain tumor detection system achieving 97%+ accuracy
>  with Grad-CAM explainability, deployed as a live Streamlit web application."
