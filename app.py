# ============================================================
#  BRAIN TUMOR DETECTION — STREAMLIT APP
# ============================================================

import os
import subprocess
import sys
import gdown


# Force install gdown if not available
# try:
#     import gdown
# except ImportError:
#     # subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown==4.7.3"])
#     import gdown

# Force install gdown if not available
try:
    import gdown
except ImportError:
    import gdown  # will be installed via requirements.txt
    
import numpy as np
import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor Detector",
    page_icon="🧠",
    layout="wide"
)

# ── SETTINGS ────────────────────────────────────────────────
CLASSES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

CLASS_INFO = {
    'Glioma': {
        'color': '#FF6B6B',
        'desc': 'Glioma is a tumor that starts in the glial cells of the brain or spine.',
        'severity': '🔴 High Risk'
    },
    'Meningioma': {
        'color': '#FFA94D',
        'desc': 'Meningioma arises from the meninges — membranes surrounding the brain.',
        'severity': '🟡 Medium Risk'
    },
    'No Tumor': {
        'color': '#51CF66',
        'desc': 'No tumor detected. The brain scan appears normal.',
        'severity': '🟢 Normal'
    },
    'Pituitary': {
        'color': '#339AF0',
        'desc': 'Pituitary tumor is an abnormal growth in the pituitary gland.',
        'severity': '🟠 Medium Risk'
    }
}

# ── DOWNLOAD MODEL FROM GOOGLE DRIVE ────────────────────────
FILE_ID    = "YOUR_GOOGLE_DRIVE_FILE_ID"   # 👈 yahan taro FILE_ID nakhje
MODEL_PATH = "brain_tumor_vgg16_final.keras"

if not os.path.exists(MODEL_PATH):
    with st.spinner("📥 Downloading model from Google Drive..."):
        gdown.download(
            f"https://drive.google.com/uc?id={FILE_ID}",
            MODEL_PATH,
            quiet=False
        )

# ── LOAD MODEL ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ── GRAD-CAM ─────────────────────────────────────────────────
def get_gradcam(model, img_array, last_conv_layer='block5_conv3'):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_class   = tf.argmax(predictions[0])
        class_output = predictions[:, pred_class]

    grads        = tape.gradient(class_output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap      = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(heatmap)
    heatmap      = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_gradcam(original_img, heatmap):
    img             = np.array(original_img.resize((224, 224)))
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    superimposed    = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)
    return superimposed

# ── PREPROCESS ───────────────────────────────────────────────
def preprocess(image):
    img = image.resize((224, 224)).convert('RGB')
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.vgg16.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

# ── UI ───────────────────────────────────────────────────────
st.title("🧠 Brain Tumor MRI Detection")
st.markdown("AI-powered detection using **VGG16 Transfer Learning** with **Grad-CAM** explainability")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.info("""
    This app detects brain tumors from MRI scans using deep learning.

    **Model:** VGG16 Transfer Learning
    **Classes:** Glioma, Meningioma, No Tumor, Pituitary
    **Input:** Brain MRI Scan (JPG/PNG)
    """)
    st.header("📊 Class Guide")
    for cls, info in CLASS_INFO.items():
        st.markdown(f"**{cls}** — {info['severity']}")
    st.markdown("---")
    st.warning("⚠️ For research purposes only. Not a substitute for clinical diagnosis.")

# Upload
uploaded = st.file_uploader(
    "Upload Brain MRI Scan",
    type=['jpg', 'jpeg', 'png'],
    help="Upload a brain MRI image for tumor detection"
)

if uploaded is not None:
    image     = Image.open(uploaded).convert('RGB')
    img_array = preprocess(image)

    with st.spinner("🔍 Analyzing MRI scan..."):
        preds       = model.predict(img_array)[0]
        pred_idx    = np.argmax(preds)
        pred_class  = CLASSES[pred_idx]
        confidence  = preds[pred_idx] * 100
        heatmap     = get_gradcam(model, img_array)
        gradcam_img = overlay_gradcam(image, heatmap)

    st.markdown("---")

    # Result Banner
    info = CLASS_INFO[pred_class]
    st.markdown(
        f"""
        <div style='background-color:{info["color"]}22;
                    border-left: 5px solid {info["color"]};
                    padding: 1rem; border-radius: 8px; margin-bottom: 1rem;'>
            <h3 style='color:{info["color"]}; margin:0;'>
                {info["severity"]} — {pred_class}
            </h3>
            <p style='margin:0.5rem 0 0;'>{info["desc"]}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 3 Columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📷 Original MRI")
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("🔥 Grad-CAM Heatmap")
        st.image(gradcam_img, use_column_width=True)
        st.caption("Red regions = areas model focused on")

    with col3:
        st.subheader("📊 Confidence Scores")
        st.metric(
            label="Prediction",
            value=pred_class,
            delta=f"{confidence:.1f}% confidence"
        )
        st.markdown("**All class probabilities:**")
        for i, (cls, prob) in enumerate(zip(CLASSES, preds)):
            bar_color = "🔴" if i == pred_idx else "⚪"
            st.markdown(f"{bar_color} **{cls}**")
            st.progress(float(prob))
            st.caption(f"{prob * 100:.2f}%")

    # Top 2 Predictions
    st.markdown("---")
    st.subheader("🏆 Top Predictions")
    top2_idx = np.argsort(preds)[::-1][:2]
    c1, c2   = st.columns(2)

    for i, (col, idx) in enumerate(zip([c1, c2], top2_idx)):
        with col:
            st.markdown(
                f"""
                <div style='background:{CLASS_INFO[CLASSES[idx]]["color"]}22;
                            border:1px solid {CLASS_INFO[CLASSES[idx]]["color"]};
                            padding:1rem; border-radius:8px; text-align:center;'>
                    <h4>#{i+1} {CLASSES[idx]}</h4>
                    <h2 style='color:{CLASS_INFO[CLASSES[idx]]["color"]};'>
                        {preds[idx]*100:.1f}%
                    </h2>
                </div>
                """,
                unsafe_allow_html=True
            )

else:
    st.markdown("""
    <div style='text-align:center; padding: 3rem;
                border: 2px dashed #ccc; border-radius: 12px;'>
        <h3>👆 Upload an MRI scan to get started</h3>
        <p style='color:gray;'>Supported formats: JPG, JPEG, PNG</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🎯 How it works")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 1️⃣ Upload\nUpload a brain MRI scan image")
    with c2:
        st.markdown("### 2️⃣ Analyze\nVGG16 model analyzes the scan")
    with c3:
        st.markdown("### 3️⃣ Result\nGet prediction + Grad-CAM heatmap")
