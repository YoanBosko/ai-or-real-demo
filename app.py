from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


st.set_page_config(
    page_title="AI or Real? | Image Classifier",
    page_icon="🔎",
    layout="centered",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = Path(__file__).parent / "image_classification_cnn.h5"
IMAGE_SIZE = (150, 150)
# flow_from_directory() in CNN.ipynb assigns classes alphabetically by folder name.
CLASS_LABELS = ["ManusiaAI", "ManusiaReal", "PemandanganAI", "PemandanganReal"]
LABEL_DETAILS = {
    "ManusiaAI": ("Manusia", "Dibuat AI"),
    "ManusiaReal": ("Manusia", "Foto nyata"),
    "PemandanganAI": ("Pemandangan", "Dibuat AI"),
    "PemandanganReal": ("Pemandangan", "Foto nyata"),
}


@st.cache_resource(show_spinner="Memuat model klasifikasi...")
def load_model(model_path: str, modified_at: float):
    """Load once per app process; modified_at refreshes the cache after model updates."""
    del modified_at
    return tf.keras.models.load_model(model_path, compile=False)


def preprocess_image(image_file) -> tuple[Image.Image, np.ndarray]:
    """Match CNN.ipynb preprocessing: RGB, 150x150, then divide pixels by 255."""
    image = Image.open(image_file).convert("RGB")
    resized_image = image.resize(IMAGE_SIZE)
    image_array = np.asarray(resized_image, dtype=np.float32) / 255.0
    return image, np.expand_dims(image_array, axis=0)


def format_confidence(confidence: float) -> str:
    return f"{confidence * 100:.1f}%"


st.markdown(
    """
    <style>
      .stApp { background: #fff9f1; color: #172554; }
      .stApp p, .stApp label, .stApp li, .stApp small,
      .stApp [data-testid="stFileUploaderDropzoneInstructions"],
      .stApp [data-testid="stFileUploaderDropzoneInstructions"] span {
        color: #172554 !important;
      }
      .hero { text-align: center; padding: 1.4rem 0 0.7rem; }
      .hero h1 { color: #9f204d; margin-bottom: 0.35rem; }
      .hero p { color: #1e3a8a; font-size: 1.05rem; }
      .result-card {
        background: #ffffff; border: 1px solid #f0ced9; border-radius: 16px;
        padding: 1.2rem 1.4rem; text-align: center; margin: 1rem 0;
      }
      .result-card h2 { color: #9f204d; margin: 0.2rem 0; }
      .result-card p { margin: 0.2rem 0; color: #1e3a8a; }
    </style>
    <div class="hero">
      <h1>AI or Real? 🔎</h1>
      <p>Unggah gambar manusia atau pemandangan untuk mendapat prediksi dari model CNN.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Tentang model")
    st.write("CNN ini membedakan manusia/pemandangan serta gambar AI/foto nyata.")
    st.caption("Input diproses sebagai RGB berukuran 150 × 150 piksel.")

uploaded_file = st.file_uploader(
    "Pilih gambar",
    type=["jpg", "jpeg", "png"],
    help="Model dilatih terutama dengan gambar JPG manusia dan pemandangan.",
)

if uploaded_file is None:
    st.info("Mulai dengan mengunggah satu gambar JPG, JPEG, atau PNG.")
    st.stop()

try:
    original_image, input_tensor = preprocess_image(uploaded_file)
except Exception:
    st.error("Gambar tidak dapat dibaca. Pilih file gambar JPG, JPEG, atau PNG yang valid.")
    st.stop()

st.image(original_image, caption="Gambar yang diunggah", use_container_width=True)

if not MODEL_PATH.exists():
    st.error("File model `image_classification_cnn.h5` tidak ditemukan di repository.")
    st.stop()

try:
    model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
    probabilities = model.predict(input_tensor, verbose=0)[0]
except Exception as error:
    st.error("Model gagal dimuat atau menjalankan prediksi.")
    st.exception(error)
    st.stop()

best_index = int(np.argmax(probabilities))
best_label = CLASS_LABELS[best_index]
subject, origin = LABEL_DETAILS[best_label]

st.markdown(
    f"""
    <div class="result-card">
      <p>Prediksi model</p>
      <h2>{origin}: {subject}</h2>
      <p>Tingkat keyakinan: <strong>{format_confidence(float(probabilities[best_index]))}</strong></p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("Rincian prediksi")
for label, probability in zip(CLASS_LABELS, probabilities):
    subject_name, origin_name = LABEL_DETAILS[label]
    st.write(f"**{origin_name} — {subject_name}** · {format_confidence(float(probability))}")
    st.progress(float(probability))

st.caption(
    "Prediksi adalah hasil model eksperimen, bukan bukti pasti bahwa suatu gambar dibuat AI atau merupakan foto nyata."
)
