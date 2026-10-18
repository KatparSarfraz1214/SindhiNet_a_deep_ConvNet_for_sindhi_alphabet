import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "best_sindhi_ocr_model.keras"
CLASS_NAMES_PATH = APP_DIR / "class_names.json"
IMAGE_SIZE = (64, 64)

st.set_page_config(
    page_title="SindhiNet | Character Recognition",
    page_icon="🔤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Polished UI styling ----------
st.markdown(
    """
    <style>
    :root {
        --green: #16834a;
        --green-dark: #0d6035;
        --green-pale: #eaf7ef;
        --ink: #17251d;
        --muted: #637268;
        --line: #dce9df;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 85% 0%, rgba(22,131,74,.09), transparent 32%),
            #f6faf7;
        color: var(--ink);
    }
    [data-testid="stHeader"] { background: rgba(246,250,247,.92); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #102d20 0%, #163b2a 100%);
    }
    [data-testid="stSidebar"] * { color: #f2fbf5 !important; }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.2); }
    .block-container { padding-top: 2.1rem; padding-bottom: 2.5rem; max-width: 1250px; }
    .hero {
        background: linear-gradient(115deg, #103d28 0%, #167746 60%, #27965a 100%);
        padding: 1.8rem 2rem;
        border-radius: 22px;
        color: white;
        margin-bottom: 1.35rem;
        box-shadow: 0 12px 32px rgba(17, 88, 48, .15);
    }
    .hero h1 { color: white; font-size: 2.45rem; margin: 0; letter-spacing: -.04em; }
    .hero p { color: #e4f5e9; margin: .45rem 0 0 0; font-size: 1.04rem; }
    .eyebrow { color: #c3efd1; text-transform: uppercase; font-weight: 800; font-size: .72rem; letter-spacing: .15em; margin-bottom: .55rem; }
    .section-title { font-size: 1.25rem; font-weight: 800; color: #163b27; margin: .25rem 0 .7rem 0; }
    .soft-card {
        background: white; border: 1px solid var(--line); border-radius: 17px;
        padding: 1.1rem 1.2rem; box-shadow: 0 4px 16px rgba(21, 61, 36, .045);
    }
    .prediction-card {
        background: linear-gradient(145deg, #ffffff 0%, #effaf2 100%);
        border: 1px solid #cdebd6; border-radius: 20px; padding: 1.5rem;
        text-align: center; min-height: 210px;
    }
    .prediction-label { color: #607568; text-transform: uppercase; letter-spacing: .12em; font-size: .75rem; font-weight: 800; }
    .prediction-character { color: #12683a; font-size: 5rem; line-height: 1.25; font-weight: 800; margin: .35rem 0; }
    .prediction-note { color: #5b6c61; font-size: .9rem; }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #147341, #1b9452); border: 0;
        border-radius: 11px; min-height: 3rem; font-weight: 750;
        box-shadow: 0 5px 13px rgba(22,131,74,.18);
    }
    div.stButton > button[kind="primary"]:hover { background: #105d34; border: 0; }
    [data-testid="stFileUploader"] {
        background: white; border: 1px dashed #9ac9a9; border-radius: 16px; padding: .6rem;
    }
    [data-testid="stMetric"] {
        background: white; border: 1px solid var(--line); border-radius: 13px; padding: .8rem;
    }
    .small-muted { color: #65766b; font-size: .88rem; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Model and preprocessing ----------
@st.cache_resource(show_spinner="Loading SindhiNet model...")
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MODEL_PATH.name}. Put the trained model beside app.py."
        )
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CLASS_NAMES_PATH.name}. Put the class-label file beside app.py."
        )
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    if not isinstance(class_names, list):
        raise ValueError("class_names.json must contain a JSON list.")
    if len(class_names) != int(model.output_shape[-1]):
        raise ValueError(
            f"Found {len(class_names)} labels but model outputs "
            f"{model.output_shape[-1]} classes. Check the label order and model."
        )
    return model, class_names


def preprocess_for_model(image: Image.Image) -> np.ndarray:
    """Match training input: grayscale, resize to 64x64, normalize to [0, 1]."""
    image = ImageOps.exif_transpose(image).convert("L")
    image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array[np.newaxis, ..., np.newaxis]


def pretty_label(label):
    return str(label)


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🔤 SindhiNet")
    st.markdown("**Handwritten alphabet recognition**")
    st.divider()
    st.markdown("### Model details")
    st.markdown("**Architecture**  \nResidual CNN")
    st.markdown("**Input size**  \n64 × 64 grayscale")
    st.markdown("**Output classes**  \n52 Sindhi characters")
    st.divider()
    st.markdown("### Tips for best results")
    st.markdown(
        "- Upload one character at a time.\n"
        "- Use a sharp, well-lit image.\n"
        "- Keep the character centered.\n"
        "- Avoid full words or busy backgrounds."
    )
    st.divider()
    st.caption("SindhiNet · Sarfraz Ali Katpar")

# ---------- Header ----------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">A Deep Convolutipnal Network for Sindhi Handwritten Character Recognition</div>
      <h1>SindhiNet</h1>
      <p>Turn a handwritten Sindhi character into a model prediction in seconds.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    model, class_names = load_artifacts()
except Exception as exc:
    st.error(f"Unable to load SindhiNet: {exc}")
    st.info("Check that the model and class_names.json are in the same folder as app.py.")
    st.stop()

# Quick overview metrics
m1, m2, m3 = st.columns(3)
m1.metric("Character classes", len(class_names))
m2.metric("Model input", "64 × 64")
m3.metric("Image format", "Grayscale")

st.write("")
st.markdown('<div class="section-title">1 · Upload your character image</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Choose a character image",
    type=["png", "jpg", "jpeg", "bmp", "webp"],
    help="For the best result, upload one handwritten character rather than a full word.",
    label_visibility="collapsed",
)

if uploaded_file is None:
    st.markdown(
        """
        <div class="soft-card">
          <h3 style="margin-top:0;color:#173e29">Ready when you are</h3>
          <p style="color:#65766b;margin-bottom:.6rem">
            Upload a PNG, JPG, JPEG, BMP, or WEBP image. You'll be able to inspect
            the original, preview the model input, and compare the strongest predictions.
          </p>
          <span style="background:#eaf7ef;color:#17663a;padding:.35rem .65rem;border-radius:999px;font-size:.82rem;font-weight:700">
            One image · One character · Instant prediction
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    try:
        original = Image.open(uploaded_file)
        original = ImageOps.exif_transpose(original)
        preview = original.convert("RGB")
        ready = original.convert("L").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)

        st.markdown('<div class="section-title">2 · Review the image</div>', unsafe_allow_html=True)
        left, right = st.columns([1, 1], gap="large")
        with left:
            st.markdown('<div class="soft-card">', unsafe_allow_html=True)
            st.markdown("**Original image**")
            st.image(preview, use_container_width=True)
            st.caption(f"{original.width} × {original.height} pixels")
            st.markdown('</div>', unsafe_allow_html=True)
        with right:
            st.markdown('<div class="soft-card">', unsafe_allow_html=True)
            st.markdown("**Model input preview**")
            st.image(ready, use_container_width=True)
            st.caption("Grayscale · resized to 64 × 64 · pixel values normalized to 0–1")
            st.markdown('</div>', unsafe_allow_html=True)

        run_col, clear_col = st.columns([3, 1])
        with run_col:
            run_prediction = st.button(
                "✨  Recognize character",
                type="primary",
                use_container_width=True,
            )
        with clear_col:
            clear_prediction = st.button("Reset", use_container_width=True)

        if clear_prediction:
            st.session_state.pop("sindhinet_prediction", None)
            st.rerun()

        if run_prediction:
            with st.spinner("SindhiNet is analyzing the character..."):
                x = preprocess_for_model(original)
                probabilities = np.asarray(model.predict(x, verbose=0)[0])
                if probabilities.ndim != 1 or len(probabilities) != len(class_names):
                    st.error("Unexpected model output shape. Please check the model and labels.")
                    st.stop()
                top_indices = np.argsort(probabilities)[::-1][: min(5, len(class_names))]
                st.session_state["sindhinet_prediction"] = {
                    "filename": uploaded_file.name,
                    "best_idx": int(top_indices[0]),
                    "probabilities": probabilities.tolist(),
                    "top_indices": [int(i) for i in top_indices],
                }

        result = st.session_state.get("sindhinet_prediction")
        if result and result.get("filename") == uploaded_file.name:
            probabilities = np.asarray(result["probabilities"], dtype=float)
            best_idx = int(result["best_idx"])
            best_label = pretty_label(class_names[best_idx])
            confidence = float(probabilities[best_idx]) * 100

            st.markdown('<div class="section-title">3 · Recognition results</div>', unsafe_allow_html=True)
            out_left, out_right = st.columns([1, 1.25], gap="large")
            with out_left:
                st.markdown(
                    f"""
                    <div class="prediction-card">
                      <div class="prediction-label">Predicted character</div>
                      <div class="prediction-character" dir="auto">{best_label}</div>
                      <div class="prediction-note">Highest scoring class</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.metric("Model confidence", f"{confidence:.2f}%")
            with out_right:
                st.markdown('<div class="soft-card">', unsafe_allow_html=True)
                st.markdown("**Top 5 candidate classes**")
                st.caption("Compare the model's scores before interpreting the result.")
                for rank, idx in enumerate(result["top_indices"], start=1):
                    label = pretty_label(class_names[int(idx)])
                    score = float(probabilities[int(idx)])
                    row_a, row_b = st.columns([1, 3])
                    with row_a:
                        st.markdown(
                            f"<div style='font-size:1.35rem;font-weight:800;color:#17663a' dir='auto'>{rank}. {label}</div>",
                            unsafe_allow_html=True,
                        )
                    with row_b:
                        st.progress(float(np.clip(score, 0.0, 1.0)))
                        st.caption(f"{score * 100:.2f}%")
                st.markdown('</div>', unsafe_allow_html=True)

            with st.expander("View all 52 class scores"):
                sorted_indices = np.argsort(probabilities)[::-1]
                for idx in sorted_indices:
                    st.write(f"{class_names[int(idx)]} — {float(probabilities[int(idx)]) * 100:.3f}%")

            st.warning(
                "These values are model softmax scores, not guarantees of correctness. "
                "Image quality and similarity to training examples can affect the result."
            )
        else:
            st.markdown(
                '<div class="small-muted">Click <b>Recognize character</b> to run the model.</div>',
                unsafe_allow_html=True,
            )

    except Exception as exc:
        st.error(f"Could not process this image: {exc}")

st.divider()
with st.expander("About this project"):
    st.write(
        "SindhiNet is a deep convolutional neural network for recognizing handwritten "
        "Sindhi alphabet characters. The interface converts the uploaded image to "
        "grayscale, resizes it to 64 × 64, and scales pixel values to [0, 1] before inference."
    )
st.caption("SindhiNet · Deep CNN for Sindhi Handwritten Alphabet Recognition")
