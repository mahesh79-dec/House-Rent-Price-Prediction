from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "rent_model.joblib"
DATA_PATH = ROOT / "data" / "house_rent_data.csv"

st.set_page_config(page_title="House Rent Price Predictor", page_icon="🏠", layout="centered")
st.title("🏠 House Rent Price Prediction")
st.write("Estimate monthly house rent using a machine-learning model.")
st.warning(
    "This project currently uses a synthetic demonstration dataset. "
    "Predictions are for academic demonstration only and are not verified market quotes."
)

if not MODEL_PATH.exists():
    st.info("The model has not been trained yet. Training it now…")
    from train_model import main
    main()

model = joblib.load(MODEL_PATH)
data = pd.read_csv(DATA_PATH)

cities = sorted(data["City"].unique())
city = st.selectbox("City", cities)
available_localities = sorted(data.loc[data["City"] == city, "Locality"].unique())
locality = st.selectbox("Locality", available_localities)
col1, col2 = st.columns(2)
with col1:
    bhk = st.selectbox("Bedrooms (BHK)", [1, 2, 3, 4])
    bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4])
    size = st.slider("House size (sq. ft.)", 250, 2200, 850, 50)
with col2:
    furnishing = st.selectbox("Furnishing", ["Unfurnished", "Semi-furnished", "Furnished"])
    property_type = st.selectbox("Property type", ["Apartment", "Independent House", "Studio"])
    floor = st.slider("Floor number", 0, 20, 2)

if st.button("Predict Monthly Rent", type="primary", use_container_width=True):
    sample = pd.DataFrame([{
        "City": city, "Locality": locality, "BHK": bhk, "Size_sqft": size,
        "Bathrooms": bathrooms, "Furnishing": furnishing,
        "Property_Type": property_type, "Floor": floor
    }])
    prediction = max(0, float(model.predict(sample)[0]))
    st.success(f"Estimated monthly rent: ₹{prediction:,.0f}")
    st.caption("This is a model estimate, not a guaranteed or current market price.")

with st.expander("About this project"):
    st.write(
        "A Random Forest Regression pipeline learns patterns from property features. "
        "Categorical inputs are one-hot encoded; numerical inputs are passed through. "
        "The model is evaluated using MAE, RMSE and R² in train_model.py."
    )
