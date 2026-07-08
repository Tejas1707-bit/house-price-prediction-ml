import streamlit as st
import pandas as pd
import joblib

# ---------- Page setup ----------
st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏠",
    layout="centered",
)

# ---------- Load model (cached so it only loads once) ----------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# Exact column order the model was trained on.
# 8 numeric columns + 5 one-hot columns for ocean_proximity.
FEATURE_COLUMNS = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "ocean_proximity_<1H OCEAN",
    "ocean_proximity_INLAND",
    "ocean_proximity_ISLAND",
    "ocean_proximity_NEAR BAY",
    "ocean_proximity_NEAR OCEAN",
]

OCEAN_PROXIMITY_OPTIONS = [
    "<1H OCEAN",
    "INLAND",
    "ISLAND",
    "NEAR BAY",
    "NEAR OCEAN",
]

# ---------- Header ----------
st.title("🏠 California House Price Predictor")
st.write(
    "Estimate median house value for a California district using a "
    "Random Forest Regressor trained on the classic California Housing dataset."
)

st.divider()

# ---------- Input form ----------
st.subheader("Enter district details")

col1, col2 = st.columns(2)

with col1:
    longitude = st.number_input(
        "Longitude", min_value=-124.5, max_value=-114.0, value=-119.5, step=0.01,
        help="Geographic longitude of the district (California ranges roughly -124.5 to -114.0)"
    )
    housing_median_age = st.number_input(
        "Housing Median Age (years)", min_value=1, max_value=100, value=25, step=1
    )
    total_bedrooms = st.number_input(
        "Total Bedrooms", min_value=1, max_value=10000, value=500, step=1
    )
    households = st.number_input(
        "Households", min_value=1, max_value=10000, value=400, step=1
    )

with col2:
    latitude = st.number_input(
        "Latitude", min_value=32.5, max_value=42.0, value=36.5, step=0.01,
        help="Geographic latitude of the district (California ranges roughly 32.5 to 42.0)"
    )
    total_rooms = st.number_input(
        "Total Rooms", min_value=1, max_value=50000, value=2500, step=1
    )
    population = st.number_input(
        "Population", min_value=1, max_value=50000, value=1200, step=1
    )
    median_income = st.number_input(
        "Median Income (in tens of thousands, e.g. 5.0 = $50,000)",
        min_value=0.0, max_value=20.0, value=4.0, step=0.1
    )

ocean_proximity = st.selectbox("Ocean Proximity", OCEAN_PROXIMITY_OPTIONS, index=0)

st.divider()

# ---------- Prediction ----------
if st.button("Predict House Price", type="primary", use_container_width=True):
    # Build the one-hot encoded row in the exact order the model expects
    row = {
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
    }
    for option in OCEAN_PROXIMITY_OPTIONS:
        row[f"ocean_proximity_{option}"] = 1 if ocean_proximity == option else 0

    input_df = pd.DataFrame([row])[FEATURE_COLUMNS]

    prediction = model.predict(input_df)[0]

    st.success(f"### Estimated Median House Value: **${prediction:,.2f}**")

    with st.expander("See the exact input sent to the model"):
        st.dataframe(input_df.T.rename(columns={0: "value"}))

    # Show the location on a map
    st.subheader("District Location")
    st.map(pd.DataFrame({"lat": [latitude], "lon": [longitude]}), zoom=5)

st.divider()
st.caption(
    "Model: scikit-learn RandomForestRegressor | Dataset: California Housing Prices"
)
