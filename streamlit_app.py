import streamlit as st
import pandas as pd
import shap
import lightgbm as lgb
import joblib
from streamlit_shap import st_shap

# Title
st.title("🧠 Global Feature Impact (SHAP)")

# Load model and data
model = joblib.load("model.pkl")  # Your LightGBM model
X = pd.read_csv("input.csv")      # Input data

# Compute SHAP values
explainer = shap.Explainer(model)
shap_values = explainer(X)

# SHAP values (n_samples, n_features) expected for summary
shap_array = shap_values.values  # Extract underlying array

# Debug info
st.code(f"SHAP shape: {shap_array.shape}")
st.code(f"Input shape: {X.shape}")

# 🔍 Global summary plot (only if SHAP shape matches input shape)
st.subheader("🧠 Global Feature Impact (SHAP)")
if shap_array.shape == X.shape:
    st_shap(shap.plots.beeswarm(shap_values, max_display=10), height=400)
else:
    st.warning("⚠️ SHAP value shape mismatch. Cannot plot summary.")

# 🔬 Record-Level Force Plot
st.subheader("🔍 Record-Level SHAP Force Plot")
for i in range(min(5, len(X))):
    st.markdown(f"**Record {i+1}**")
    try:
        st_shap(shap.plots.force(explainer.expected_value, shap_array[i], X.iloc[i]), height=300)
    except Exception as e:
        st.warning(f"⚠️ Could not render force plot for record {i+1}: {e}")
