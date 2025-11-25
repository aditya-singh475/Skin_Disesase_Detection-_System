import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageFilter
import numpy as np
import pandas as pd
import json
import altair as alt
import io

# Load model and class map
@st.cache_resource
def load_model_and_map():
    model = load_model("model/models/skin_model.h5")
    with open("model/models/class_indices.json") as f:
        class_indices = json.load(f)
    inv_map = {v: k for k, v in class_indices.items()}
    return model, inv_map

model, inv_map = load_model_and_map()

# Preprocess function
def preprocess(img, size=(224, 224)):
    img = img.resize(size)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# Confidence style
def confidence_style(conf):
    if conf >= 0.85:
        return "#00ff00", "High certainty ✅"
    elif conf >= 0.5:
        return "#ffcc00", "Moderate certainty ℹ️"
    else:
        return "#ff4444", "Low certainty ⚠️ Please verify manually."

# Class descriptions
descriptions = {
    "fungal": "Fungal infections often appear as itchy, red patches. Common types include ringworm and candidiasis.",
    "acne": "Acne includes pimples, blackheads, and inflammation. Often caused by clogged pores.",
    "melanoma": "Melanoma is a serious type of skin cancer that develops in pigment-producing cells.",
    "rosacea": "Rosacea causes redness and visible blood vessels in your face, sometimes with small bumps.",
    "eczema": "Eczema makes your skin red and itchy. It's common in children but can occur at any age."
}

# UI config
st.set_page_config(page_title="Skin Health Detection", layout="centered")
st.title("🧠 Skin Health Detection Demo")

# Upload multiple images
uploaded_files = st.file_uploader("Upload multiple skin images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    enhance = st.checkbox("🔧 Enhance images before prediction (optional)")
    results = []

    for uploaded in uploaded_files:
        img = Image.open(uploaded).convert("RGB")
        if enhance:
            img = img.filter(ImageFilter.SHARPEN)

        st.image(img, caption=uploaded.name, use_container_width=True)

        arr = preprocess(img)
        pred = model.predict(arr)
        cls_id = int(np.argmax(pred))
        conf = float(np.max(pred))
        cls_name = inv_map[cls_id]
        color, label = confidence_style(conf)

        st.markdown(
            f"""
            <div style='background-color:#1a1a1a; padding:15px; border-radius:10px; text-align:center;'>
                <span style='color:{color}; font-size:22px; font-weight:bold;'>📊 Confidence: {conf:.4f}</span><br>
                <span style='color:{color}; font-size:16px;'>{label}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.progress(conf)

        if cls_name in descriptions:
            st.markdown(
                f"""
                <div style='background-color:#333; padding:15px; border-radius:10px; margin-top:15px;'>
                    <h4 style='color:#ffcc00;'>📘 About {cls_name}</h4>
                    <p style='color:#ffffff; font-size:16px;'>{descriptions[cls_name]}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        results.append({
            "filename": uploaded.name,
            "predicted_class": cls_name,
            "confidence": conf
        })

    # Show results table
    df = pd.DataFrame(results)
    st.subheader("📁 Prediction Results")
    st.dataframe(df, use_container_width=True)

    # Download CSV
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    st.download_button("Download CSV", csv_buf.getvalue(), file_name="multi_predictions.csv", mime="text/csv")

    # Charts
    st.subheader("📊 Class Distribution")
    cls_counts = df["predicted_class"].value_counts().reset_index()
    cls_counts.columns = ["class", "count"]
    chart = alt.Chart(cls_counts).mark_bar().encode(
        x=alt.X("class:N", sort="-y"),
        y=alt.Y("count:Q"),
        color=alt.Color("class:N", scale=alt.Scale(scheme="category10")),
        tooltip=["class", "count"]
    ).properties(height=300, background="#0e1117")
    st.altair_chart(chart, use_container_width=True)

    st.subheader("📈 Confidence Distribution")
    conf_chart = alt.Chart(df).mark_area(
        line={"color": "#1f77b4"},
        color="#1f77b4",
        opacity=0.25
    ).encode(
        x=alt.X("confidence:Q", bin=alt.Bin(maxbins=30)),
        y=alt.Y("count()"),
        tooltip=["confidence"]
    ).properties(height=300, background="#0e1117")
    st.altair_chart(conf_chart, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888; font-size:14px;'>"
    "🔬 Powered by CNN Model | Streamlit Demo crafted with care by <b>Aditya</b>"
    "</div>",
    unsafe_allow_html=True
)


