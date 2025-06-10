import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import os

# -------------------- Load Models ---------------------
# Load flight price model and scaler
with open("flight_price/rf_model.pkl", "rb") as f:
    flight_model = pickle.load(f)
with open("flight_price/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load hotel recommender model
from hotel_recommendation.recommender import CFRecommender

with open("hotel_recommendation/cf_recommender.pkl", "rb") as f:
    cf_recommender_model = pickle.load(f)


# -------------------- Flight Feature Order ---------------------
feature_order = [
    "from_Florianopolis (SC)", "from_Sao_Paulo (SP)", "from_Salvador (BH)", "from_Brasilia (DF)", "from_Rio_de_Janeiro (RJ)", "from_Campo_Grande (MS)",
    "from_Aracaju (SE)", "from_Natal (RN)", "from_Recife (PE)", 
    "destination_Florianopolis (SC)", "destination_Sao_Paulo (SP)", "destination_Salvador (BH)", "destination_Brasilia (DF)", "destination_Rio_de_Janeiro (RJ)", "destination_Campo_Grande (MS)",
    "destination_Aracaju (SE)", "destination_Natal (RN)", "destination_Recife (PE)", 
    "flightType_economic", "flightType_firstClass", "flightType_premium", 
    "agency_Rainbow", "agency_CloudFy", "agency_FlyingDrops", 
    "month", "year", "day"
]

# -------------------- Streamlit UI ---------------------
st.title("✈️ Travel Planner Dashboard")

tab1, tab2 = st.tabs(["Flight Price Prediction", "Hotel Recommendation"])

# -------------------- Tab 1: Flight Price Prediction ---------------------
with tab1:
    st.header("Flight Price Predictor")

    Departure = st.selectbox("Departure City", sorted([x.split("from_")[1] for x in feature_order if x.startswith("from_")]))
    Destination = st.selectbox("Destination City", sorted([x.split("destination_")[1] for x in feature_order if x.startswith("destination_")]))
    FlightType = st.selectbox("Flight Type", ["economic", "firstClass", "premium"])
    Agency = st.selectbox("Agency", ["Rainbow", "CloudFy", "FlyingDrops"])
    col1, col2, col3 = st.columns(3)
    with col1:
        month = st.number_input("Month", min_value=1, max_value=12, step=1)
    with col2:
        day = st.number_input("Day", min_value=1, max_value=31, step=1)
    with col3:
        year = st.number_input("Year", min_value=2023, max_value=2030, step=1)

    if st.button("Predict Flight Price"):
        try:
            input_features = np.zeros(len(feature_order))
            if f"from_{Departure}" in feature_order:
                input_features[feature_order.index(f"from_{Departure}")] = 1
            if f"destination_{Destination}" in feature_order:
                input_features[feature_order.index(f"destination_{Destination}")] = 1
            if f"flightType_{FlightType}" in feature_order:
                input_features[feature_order.index(f"flightType_{FlightType}")] = 1
            if f"agency_{Agency}" in feature_order:
                input_features[feature_order.index(f"agency_{Agency}")] = 1
            input_features[feature_order.index("month")] = month
            input_features[feature_order.index("year")] = year
            input_features[feature_order.index("day")] = day

            scaled_input = scaler.transform([input_features])
            prediction = flight_model.predict(scaled_input)[0]

            st.success(f"Estimated Flight Price: ${round(prediction, 2)}")


        except Exception as e:
            st.error(f"Error: {e}")

# -------------------- Tab 2: Hotel Recommendation ---------------------
with tab2:
    st.header("Hotel Recommendation")

    city_list = sorted(cf_recommender_model.items_df["place"].unique())
    selected_city = st.selectbox("Select a City", city_list)
    num_days = st.number_input("Enter Number of Days", min_value=1, max_value=30, step=1)
    budget = st.number_input("Enter Maximum Price per Day", min_value=1.0, max_value=10000.0, step=10.0)
    top_n = st.slider("Number of Hotel Recommendations", 1, 10, 5)

    if st.button("Get Hotel Recommendations"):
        recommendations = cf_recommender_model.recommend_items(selected_city, num_days, budget, topn=top_n)

        if recommendations.empty:
            st.warning("No hotels found for the given filters.")
        else:
            st.subheader("Top Hotel Options:")
            st.dataframe(recommendations)

# -------------------- Footer ---------------------
st.markdown("---")

