import streamlit as st
import numpy as np
from PIL import Image
import os

# ── Page config ──
st.set_page_config(
    page_title="FruitScan — Disease Detector",
    page_icon="🍃",
    layout="centered"
)

st.markdown("""
<style>
    .stButton>button {
        background-color: #2d6a4f;
        color: white;
        border-radius: 10px;
        font-size: 16px;
        border: none;
        width: 100%;
    }
    .stButton>button:hover { background-color: #1f4f3a; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 🍃 FruitScan — Disease Detector")
st.markdown("Upload a fruit image and the AI will detect any disease.")
st.divider()

# ── Class names (alphabetical order — same as your training folders) ──
CLASS_NAMES = [
    "Apple_Black_rot",
    "Apple_Cedar_apple_rust",
    "Apple_Healthy",
    "Apple_Scab",
    "Banana_Cordana_Leaf_Spot",
    "Banana_Healthy",
    "Banana_Panama_Disease",
    "Banana_Pestalotiopsis",
    "Cherry_Healthy",
    "Cherry_Powdery_mildew",
    "Grape_Black_Rot",
    "Grape_Healthy",
    "Grape_Leaf_blight",
    "Guava_Dot_disease",
    "Guava_Healthy",
    "Mango_Anthracnose",
    "Mango_Healthy",
    "Orange_Citrus_Greening",
    "Orange_Healthy",
    "Peach_Bacterial_spot",
    "Peach_Healthy",
    "Pomegranate_Bacterial_Blight",
    "Pomegranate_Healthy",
    "Strawberry_Healthy",
    "Strawberry_Leaf_scorch",
    "Tomato_Bacterial_Spot",
    "Tomato_Healthy",
    "Tomato_Late_blight",
]

# ── Tips ──
TIPS = {
    "healthy":     ("✅ Healthy!",       "green",  "No disease found. Keep watering regularly and monitor weekly."),
    "rot":         ("🚨 Rot",            "red",    "Remove infected fruits immediately. Apply copper-based fungicide."),
    "rust":        ("⚠️ Rust",           "orange", "Apply fungicide at bud break. Remove fallen leaves around the plant."),
    "scab":        ("⚠️ Scab",           "orange", "Improve airflow by pruning. Apply fungicide spray during wet weather."),
    "spot":        ("⚠️ Leaf Spot",      "orange", "Avoid overhead watering. Remove infected leaves. Apply fungicide."),
    "blight":      ("🚨 Blight",         "red",    "Remove affected parts immediately. Do not compost infected material."),
    "mildew":      ("⚠️ Mildew",         "orange", "Apply sulfur-based fungicide. Ensure good air circulation."),
    "greening":    ("🚨 Greening",       "red",    "No cure. Remove infected tree to prevent spreading to others."),
    "panama":      ("🚨 Panama Disease", "red",    "No cure. Uproot and destroy infected plants. Don't replant for 6 years."),
    "anthracnose": ("🚨 Anthracnose",    "red",    "Apply copper fungicide before rainy season. Remove fruit debris."),
    "default":     ("⚠️ Disease Found",  "orange", "Consult a local agronomist for a detailed treatment plan."),
}

def get_tip(label):
    for keyword, tip in TIPS.items():
        if keyword in label.lower():
            return tip
    return TIPS["default"]

# ── Load model (only once) ──
@st.cache_resource
def load_model():
    import keras
    model = keras.saving.load_model("best_model.keras")
    return model

# ── Upload ──
uploaded_file = st.file_uploader(
    "📁 Choose a fruit image",
    type=["jpg", "jpeg", "png", "webp", "bmp"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Your uploaded image", use_container_width=True)
    st.divider()

    if st.button("🔬 Analyze Fruit"):
        with st.spinner("Analyzing... please wait ⏳"):

            if not os.path.exists("best_model.keras"):
                st.error("❌ Could not find best_model.keras!\n\nMake sure it is inside your FruitScan folder.")
                st.stop()

            try:
                model = load_model()
            except Exception as e:
                st.error(f"❌ Could not load model!\n\nError: {e}")
                st.stop()

            # Prepare image — 224x224 same as training
            img = image.resize((224, 224))
            img_array = np.array(img, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            predictions = model.predict(img_array, verbose=0)[0]

            # Top 3
            top3_idx = predictions.argsort()[-3:][::-1]
            top3 = []
            for idx in top3_idx:
                label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Class {idx}"
                pct   = round(float(predictions[idx]) * 100, 1)
                top3.append((label, pct))

        best_label, best_conf = top3[0]
        icon, color, tip_text = get_tip(best_label)
        display_name = best_label.replace("_", " ")

        st.markdown("### 📊 Results")

        if color == "green":
            st.success(f"{icon}  **{display_name}**")
        elif color == "red":
            st.error(f"{icon}  **{display_name}**")
        else:
            st.warning(f"{icon}  **{display_name}**")

        st.markdown(f"**Confidence: {best_conf}%**")
        st.progress(int(best_conf))
        st.divider()

        st.markdown("**🏆 Top 3 Predictions:**")
        medals = ["🥇", "🥈", "🥉"]
        for i, (label, pct) in enumerate(top3):
            col1, col2 = st.columns([4, 1])
            col1.markdown(f"{medals[i]} {label.replace('_', ' ')}")
            col2.markdown(f"**{pct}%**")

        st.divider()
        st.markdown("**💡 What to do:**")
        st.info(tip_text)

else:
    st.markdown("""
    <div style='text-align:center; padding:3rem; color:#9ca3af;'>
        <div style='font-size:4rem;'>🍎</div>
        <p style='margin-top:1rem;'>Upload a fruit image above to get started</p>
    </div>
    """, unsafe_allow_html=True)
