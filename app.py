import streamlit as st
import numpy as np
import pickle

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌱",
    layout="centered"
)

# -------------------------------
# Load Model & Scaler
# -------------------------------
@st.cache_resource
def load_model_and_scaler():
    with open("model.pkl", "rb") as f:
        rfc, min_max = pickle.load(f)
    return rfc, min_max

rfc, min_max = load_model_and_scaler()

# -------------------------------
# Crop Dictionary
# -------------------------------
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Chickpea", 4: "Kidneybeans", 5: "Pigeonpeas",
    6: "Mothbeans", 7: "Mungbean", 8: "Blackgram", 9: "Lentil",
    10: "Pomegranate", 11: "Banana", 12: "Mango", 13: "Grapes",
    14: "Watermelon", 15: "Muskmelon", 16: "Apple", 17: "Orange",
    18: "Papaya", 19: "Coconut", 20: "Cotton", 21: "Jute", 22: "Coffee"
}

# -------------------------------
# App Title
# -------------------------------
st.title("🌱 Crop Recommendation System")
st.write("Enter soil and climate details to get the **top crop recommendations**.")

# -------------------------------
# Input Fields
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen (N)", 0, 200, 50)
    P = st.number_input("Phosphorus (P)", 0, 200, 50)
    K = st.number_input("Potassium (K)", 0, 200, 50)
    temperature = st.number_input("Temperature (°C)", value=25.0)

with col2:
    humidity = st.number_input("Humidity (%)", value=70.0)
    ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)
    rainfall = st.number_input("Rainfall (mm)", value=100.0)

# -------------------------------
# Prediction (TOP 3)
# -------------------------------
if st.button("🌾 Recommend Crops"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    input_scaled = min_max.transform(input_data)

    probabilities = rfc.predict_proba(input_scaled)[0]
    top3_indices = np.argsort(probabilities)[-3:][::-1]

    st.subheader("🌟 Top Crop Recommendations")

    for i, idx in enumerate(top3_indices, start=1):
        crop_name = crop_dict[idx + 1]
        confidence = probabilities[idx] * 100
        st.write(f"**{i}. {crop_name}** — {confidence:.2f}%")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Developed using Streamlit & Machine Learning")
