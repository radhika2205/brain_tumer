import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json, os, sys
import gdown

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gradcam.gradcam import get_gradcam_overlay
from utils.data_preprocessing import preprocess_image

# ── Google Drive Model Config ────────────────────────────────
# Step 1: best_model.h5 train karo locally
# Step 2: Google Drive pe upload karo
# Step 3: Share link thi FILE_ID copy karo
#         Link: https://drive.google.com/file/d/XXXXX/view
#                                               ^^^^^ = aa che FILE_ID
GDRIVE_FILE_ID = "YOUR_GOOGLE_DRIVE_FILE_ID"
MODEL_PATH     = "best_model.h5"
LABELS_PATH    = "model/class_indices.json"

def download_model_if_needed():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading VGG16 model... (first time only, please wait)"):
            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
        st.success("Model downloaded!")

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor Detection | VGG16",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem; font-weight: 700;
        color: #534AB7; text-align: center; margin-bottom: 0.3rem;
    }
    .subtitle {
        text-align: center; color: #888;
        font-size: 1rem; margin-bottom: 2rem;
    }
    .result-box {
        background: #f4f3ff;
        border-left: 5px solid #534AB7;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
    }
    .tumor-name { font-size: 1.8rem; font-weight: 700; }
    .tumor-color { color: #993C1D; }
    .safe-color  { color: #0F6E56; }
    .info-box {
        background: #E1F5EE; border-radius: 8px;
        padding: 0.9rem 1.1rem; font-size: 0.9rem;
        color: #085041; margin-top: 0.8rem; line-height: 1.6;
    }
    .warn-box {
        background: #FAEEDA; border-radius: 8px;
        padding: 0.9rem 1.1rem; font-size: 0.9rem;
        color: #633806; margin-top: 0.8rem;
    }
    .section-title { font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ── Load Model ────────────────────────────────────────────────
@st.cache_resource
def load_model_and_labels():
    download_model_if_needed()
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH) as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    return model, idx_to_class


# ── Tumor Info ────────────────────────────────────────────────
TUMOR_INFO = {
    'glioma': {
        'icon': '🔴',
        'desc': 'Glioma is a type of tumor that starts in the brain or spinal cord, arising from glial cells. It is one of the most common and aggressive brain tumors.',
        'severity': 'High — Immediate medical consultation required',
        'color_class': 'tumor-color'
    },
    'meningioma': {
        'icon': '🟡',
        'desc': 'Meningioma arises from the meninges — membranes surrounding the brain and spinal cord. Most are benign but can cause problems due to size.',
        'severity': 'Moderate — Medical consultation recommended',
        'color_class': 'tumor-color'
    },
    'pituitary': {
        'icon': '🔵',
        'desc': 'Pituitary tumors form in the pituitary gland at the base of the brain. Most are non-cancerous and respond well to treatment.',
        'severity': 'Low-Moderate — Specialist follow-up advised',
        'color_class': 'tumor-color'
    },
    'no_tumor': {
        'icon': '🟢',
        'desc': 'No tumor detected in the provided MRI scan. The brain tissue appears normal.',
        'severity': 'Normal — No immediate action needed',
        'color_class': 'safe-color'
    }
}


# ── Sidebar ───────────────────────────────────────────────────
def show_sidebar():
    with st.sidebar:
        st.markdown("## 🧠 About This App")
        st.markdown("""
        This app uses a **VGG16 deep CNN** trained on **7,000+ MRI scans** to detect brain tumors.

        ---
        **Detects 4 Classes:**
        - 🔴 Glioma
        - 🟡 Meningioma
        - 🔵 Pituitary Tumor
        - 🟢 No Tumor

        ---
        **Grad-CAM Explainability:**
        Heatmap overlay shows exactly which brain region the CNN focused on.
        - 🔴 Red = High attention
        - 🔵 Blue = Low attention

        ---
        **Model:** VGG16 Transfer Learning

        **Dataset:** Kaggle Brain Tumor MRI Dataset
        """)
        st.divider()
        st.caption("For educational purposes only. Not a substitute for medical diagnosis.")


# ── Main App ──────────────────────────────────────────────────
def main():
    show_sidebar()

    st.markdown('<div class="main-title">🧠 Brain Tumor Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">VGG16 CNN · Grad-CAM Explainability · 4-Class MRI Classification</div>', unsafe_allow_html=True)

    st.divider()

    uploaded_file = st.file_uploader(
        "📤 Upload a Brain MRI Scan",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a brain MRI image in JPG or PNG format"
    )

    if uploaded_file is not None:
        try:
            model, idx_to_class = load_model_and_labels()
        except Exception as e:
            st.error(f"Model load error: {e}")
            st.info("Make sure GDRIVE_FILE_ID is set correctly in streamlit_app.py and model/class_indices.json exists.")
            return

        pil_image = Image.open(uploaded_file).convert('RGB')

        with st.spinner("Analyzing MRI scan with VGG16 CNN..."):
            img_array   = preprocess_image(pil_image)
            predictions = model.predict(img_array, verbose=0)[0]
            pred_idx    = int(np.argmax(predictions))
            pred_label  = idx_to_class[pred_idx]
            confidence  = float(predictions[pred_idx]) * 100
            original_np, overlay_np, _ = get_gradcam_overlay(pil_image, model)

        st.divider()

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.markdown('<div class="section-title">📷 Original MRI</div>', unsafe_allow_html=True)
            st.image(pil_image, use_container_width=True)

        with col2:
            st.markdown('<div class="section-title">🔥 Grad-CAM Heatmap</div>', unsafe_allow_html=True)
            st.image(overlay_np, use_container_width=True)
            st.caption("Red = where VGG16 focused to make prediction")

        with col3:
            st.markdown('<div class="section-title">📊 Prediction Result</div>', unsafe_allow_html=True)
            info     = TUMOR_INFO[pred_label]
            is_tumor = pred_label != 'no_tumor'
            st.markdown(f"""
            <div class="result-box">
                <div class="tumor-name {info['color_class']}">
                    {info['icon']} {pred_label.replace('_', ' ').title()}
                </div>
                <div style="font-size:1rem; color:#555; margin-top:6px;">
                    Confidence: <b>{confidence:.1f}%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(int(confidence))
            st.markdown(f"""
            <div class="info-box"><b>About:</b> {info['desc']}</div>
            <div class="warn-box" style="margin-top:8px;"><b>Severity:</b> {info['severity']}</div>
            """, unsafe_allow_html=True)

        st.divider()
        st.markdown("### 📈 All Class Probabilities")
        class_order  = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
        prob_cols    = st.columns(4)
        class_to_idx = {v: k for k, v in idx_to_class.items()}

        for i, label in enumerate(class_order):
            idx  = class_to_idx.get(label, i)
            prob = float(predictions[idx]) * 100 if idx < len(predictions) else 0.0
            with prob_cols[i]:
                st.metric(
                    label=f"{TUMOR_INFO[label]['icon']} {label.replace('_', ' ').title()}",
                    value=f"{prob:.1f}%",
                    delta="Detected" if label == pred_label else None
                )
                st.progress(int(prob))

        st.divider()
        c1, c2, c3 = st.columns(3)
        with c1: st.info("**Model:** VGG16 Transfer Learning")
        with c2: st.info("**Explainability:** Grad-CAM (block5_conv3)")
        with c3: st.info("**Input Size:** 224 x 224 x 3")

        st.warning("This tool is for educational and research purposes only. Always consult a qualified medical professional for any diagnosis.")

    else:
        st.info("👆 Upload a brain MRI scan above to get started.")
        st.divider()
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### 🏗️ VGG16 Architecture")
            st.write("13 conv layers trained on ImageNet, fine-tuned on brain MRI data.")
        with c2:
            st.markdown("### 🔥 Grad-CAM")
            st.write("Heatmap overlay shows which brain region influenced the prediction.")
        with c3:
            st.markdown("### 🎯 4-Class Detection")
            st.write("Glioma, Meningioma, Pituitary, or No Tumor — with confidence score.")


if __name__ == "__main__":
    main()
