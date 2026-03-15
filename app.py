import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="AI House Price Estimator",
    page_icon="🏠",
    layout="wide"
)

# -----------------------------
# LOAD MODEL
# -----------------------------

model = pickle.load(open("house_price_model.pkl", "rb"))

# -----------------------------
# LOAD DATASET
# -----------------------------

@st.cache_data
def load_data():
    return pd.read_csv("cleaned_dataset.csv")

df = load_data()

# Default location value (median)
location = df["Location"].median()

# -----------------------------
# LOAD MODEL METRICS
# -----------------------------

metrics_df = pd.read_csv("model_metrics.csv")

# Automatically detect R2 column
r2_column = [col for col in metrics_df.columns if "R2" in col][0]

# Find best model
best_model_row = metrics_df.loc[metrics_df[r2_column].idxmax()]

best_model_name = best_model_row["Model"]
mae = best_model_row["MAE"]
rmse = best_model_row["RMSE"]
r2 = best_model_row[r2_column]

# -----------------------------
# TITLE
# -----------------------------

st.title("🏠 AI House Price Estimator")

st.write("Predict property prices using Machine Learning")

st.divider()

# -----------------------------
# MODEL PERFORMANCE
# -----------------------------

st.subheader("📊 Best Model Performance")

st.write(f"Best Model: **{best_model_name}**")

col1, col2, col3 = st.columns(3)

col1.metric("MAE", round(mae,3))
col2.metric("RMSE", round(rmse,3))
col3.metric("R² Score", round(r2,3))

st.divider()

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------

st.sidebar.header("🏡 Property Details")

area = st.sidebar.number_input(
    "Area (sq ft)",
    300,
    20000,
    1200
)

bedrooms = st.sidebar.slider(
    "Bedrooms",
    1,
    10,
    3
)

# City mapping
city_dict = {
    "Bangalore":0,
    "Chennai":1,
    "Delhi":2,
    "Hyderabad":3,
    "Mumbai":4
}

city_name = st.sidebar.selectbox(
    "City",
    list(city_dict.keys())
)

city = city_dict[city_name]

gym = st.sidebar.selectbox(
    "Gymnasium",
    ["No","Yes"]
)

pool = st.sidebar.selectbox(
    "Swimming Pool",
    ["No","Yes"]
)

resale = st.sidebar.selectbox(
    "Resale Property",
    ["No","Yes"]
)

# Convert Yes/No to numeric
gym = 1 if gym == "Yes" else 0
pool = 1 if pool == "Yes" else 0
resale = 1 if resale == "Yes" else 0

# -----------------------------
# PREDICTION
# -----------------------------

if st.sidebar.button("Predict Price"):

    # Create feature vector
    input_data = pd.DataFrame(
        np.zeros((1, df.shape[1]-1)),
        columns=df.drop("Price", axis=1).columns
    )

    # Fill user inputs
    input_data["Area"] = area
    input_data["Location"] = location
    input_data["No._of_Bedrooms"] = bedrooms
    input_data["Resale"] = resale
    input_data["Gymnasium"] = gym
    input_data["SwimmingPool"] = pool
    input_data["city"] = city

    # Predict log price
    prediction = model.predict(input_data)

    predicted_price = np.expm1(prediction[0])

    st.subheader("💰 Estimated Property Price")

    st.success(f"₹ {predicted_price:,.0f}")

# -----------------------------
# SHOW METRICS TABLE
# -----------------------------

st.subheader("📈 Model Evaluation Table")

st.dataframe(metrics_df)