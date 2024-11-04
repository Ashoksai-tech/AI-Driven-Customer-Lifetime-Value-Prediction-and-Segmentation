import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved model
model_path = "models/linear_regression_model.joblib"
model = joblib.load(model_path)

# Define the Streamlit app
def main():
    st.title("Customer Lifetime Value Prediction")

    # Display introduction
    st.markdown("""
    ## Welcome to the Customer Lifetime Value (CLV) Prediction App!
    This app predicts the CLV based on various customer characteristics like:
    - Recency
    - Frequency
    - Clusters (Recency, Frequency)
    - Overall Score
    
    Please enter the values below to get the predicted CLV.
    """)

    # Add user input fields with explanations
    recency = st.number_input("Recency (How recent the customer made a purchase)", min_value=0, step=1)
    frequency = st.number_input("Frequency (How often the customer makes purchases)", min_value=0, step=1)
    recency_cluster = st.number_input("Recency Cluster (Customer's recency cluster)", min_value=0, max_value=5, step=1)
    frequency_cluster = st.number_input("Frequency Cluster (Customer's frequency cluster)", min_value=0, max_value=5, step=1)
    overall_score = st.number_input("Overall Score (Calculated customer score)", min_value=0.0, max_value=10.0, step=0.1)

    # Make prediction when button is clicked
    if st.button("Predict CLV"):
        input_data = np.array([[recency, frequency, recency_cluster, frequency_cluster, overall_score]])
        clv = model.predict(input_data)[0]
        
        # Display predicted CLV
        st.subheader(f"Predicted CLV: ${clv:.2f}")
        
        # Provide a recommendation based on predicted CLV
        if clv < 50:
            st.warning("The customer is likely to have a low lifetime value. Consider implementing loyalty programs.")
        elif 50 <= clv < 150:
            st.info("The customer has a moderate lifetime value. Explore targeted promotions.")
        else:
            st.success("The customer has a high lifetime value. Focus on personalized engagement.")

if __name__ == "__main__":
    main()
