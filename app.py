import streamlit as st
import pickle
import numpy as np

# Load the trained pipeline and dataset
pipe = pickle.load(open('price_prediction_pipeline.pkl', 'rb'))
df = pickle.load(open('processed_dataset.pkl', 'rb'))

# Page config
st.set_page_config(page_title="Laptop Price Predictor", layout="centered")

# Main Title
st.title("💻 Laptop Price Predictor")
st.markdown("Estimate your laptop's price based on its specifications.")

# Section Title
st.subheader("Enter Laptop Specifications")

# 3-column layout
col1, col2, col3 = st.columns(3)

# Column Titles
with col1:
    st.markdown("### Brand & Type")
with col2:
    st.markdown("### Display & Features")
with col3:
    st.markdown("### Hardware Specs")

# Column 1 - Brand & Type
with col1:
    company = st.selectbox('Brand', df['Company'].unique())
    laptop_type = st.selectbox('Laptop Type', df['TypeName'].unique())
    os = st.selectbox('Operating System', df['os'].unique())

# Column 2 - Display & Features
with col2:
    screen_size = st.slider('Screen Size (in inches)', 10.0, 18.0, 13.3)
    resolution = st.selectbox('Screen Resolution', [
        '1920x1080', '1366x768', '1600x900', '3840x2160',
        '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'
    ])
    touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
    ips = st.selectbox('IPS Display', ['No', 'Yes'])
    weight = st.number_input('Weight (kg)', min_value=0.5, max_value=5.0, step=0.1)

# Column 3 - Hardware Specs
with col3:
    ram = st.selectbox('RAM (GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
    cpu = st.selectbox('CPU Brand', df['Cpu brand'].unique())
    gpu = st.selectbox('GPU Brand', df['Gpu brand'].unique())
    hdd = st.selectbox('HDD (GB)', [0, 128, 256, 512, 1024, 2048])
    ssd = st.selectbox('SSD (GB)', [0, 8, 128, 256, 512, 1024])

# Prediction Section
st.markdown("---")
st.subheader("Predicted Price")

if st.button('Predict Price'):
    try:
        # Encode categorical values
        touchscreen = 1 if touchscreen == 'Yes' else 0
        ips = 1 if ips == 'Yes' else 0

        # Calculate PPI
        X_res, Y_res = map(int, resolution.split('x'))
        ppi = ((X_res ** 2 + Y_res ** 2) ** 0.5) / screen_size

        # Form query
        query = np.array([
            company, laptop_type, ram, weight, touchscreen, ips, ppi,
            cpu, hdd, ssd, gpu, os
        ], dtype=object).reshape(1, -1)

        # Predict
        predicted_price = int(np.exp(pipe.predict(query)[0]))
        st.success(f"Estimated Laptop Price: ₹ {predicted_price:,}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
