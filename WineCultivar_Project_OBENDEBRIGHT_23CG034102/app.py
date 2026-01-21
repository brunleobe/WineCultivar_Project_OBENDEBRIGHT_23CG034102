import streamlit as st
import joblib
import numpy as np
import os

# 1. Page Configuration
st.set_page_config(
    page_title="Wine Cultivar Predictor",
    page_icon="🍷",
    layout="centered"
)

# 2. Artifact Loading (Cached for performance)
@st.cache_resource
def load_ml_assets():
    # Construct paths to ensure compatibility with all OS (Windows/Linux)
    model_path = os.path.join('model', 'wine_cultivar_model.pkl')
    scaler_path = os.path.join('model', 'scaler.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_ml_assets()

# 3. Header Section
st.title("🍷 Wine Cultivar Origin Prediction")
st.markdown("""
Predict the cultivar (origin/class) of wine based on its chemical properties. 
*Algorithm used: Random Forest Classifier.*
""")

# 4. Input Form
if model is None or scaler is None:
    st.error("❌ Model artifacts missing! Please run your model development script first.")
else:
    with st.form("prediction_form"):
        st.subheader("Enter Chemical Properties")
        
        # Using columns to make the UI look cleaner
        col1, col2 = st.columns(2)
        
        with col1:
            alcohol = st.number_input("Alcohol", min_value=10.0, max_value=16.0, value=13.0, step=0.1)
            flavanoids = st.number_input("Flavanoids", min_value=0.0, max_value=6.0, value=2.0, step=0.1)
            color_intensity = st.number_input("Color Intensity", min_value=0.0, max_value=15.0, value=5.0, step=0.1)
            
        with col2:
            hue = st.number_input("Hue", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
            od = st.number_input("OD280/OD315 (Diluted)", min_value=0.0, max_value=5.0, value=3.0, step=0.1)
            proline = st.number_input("Proline", min_value=200.0, max_value=2000.0, value=750.0, step=10.0)

        submit = st.form_submit_button("Predict Cultivar")

    # 5. Prediction Logic
    if submit:
        # Step 1: Prepare the 6 input features
        features = np.array([[alcohol, flavanoids, color_intensity, hue, od, proline]])
        
        # Step 2: Apply the mandatory scaling
        features_scaled = scaler.transform(features)
        
        # Step 3: Predict
        prediction = model.predict(features_scaled)[0]
        
        # Step 4: Map numeric output to human-readable text
        cultivar_map = {0: "Cultivar 1", 1: "Cultivar 2", 2: "Cultivar 3"}
        result = cultivar_map.get(prediction, "Unknown")

        # Step 5: Display Result
        st.success(f"### Result: {result}")
        st.balloons()

# 6. Sidebar Info
st.sidebar.info("System developed for Project 6: Wine Cultivar Origin Prediction.")