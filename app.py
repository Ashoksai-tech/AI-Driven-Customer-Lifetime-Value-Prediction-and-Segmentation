import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved model
model_path = "C:/Users/aasho/OneDrive/Desktop/Customer lifetime value prediction/models/linear_regression_model.joblib"
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

    # Visualize feature values as a bar chart
    st.subheader("Customer Feature Values")
    features_df = pd.DataFrame({
        "Feature": ["Recency", "Frequency", "Recency Cluster", "Frequency Cluster", "Overall Score"],
        "Value": [recency, frequency, recency_cluster, frequency_cluster, overall_score]
    })
    fig, ax = plt.subplots(figsize=(12,6))
    sns.barplot(x="Feature", y="Value", data=features_df, ax=ax)
    st.pyplot(fig)

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

    # Show additional visualizations (optional, e.g., distribution of CLV predictions)
    st.subheader("Visualizing CLV Prediction Results")
    # Example visualization of CLV prediction distribution (requires data)
    # Assuming you have some predicted values stored in 'predicted_clv_values'
    # predicted_clv_values = np.random.normal(100, 50, 100)  # Simulate CLV values for visualization
    # st.write("Sample distribution of CLV predictions:")
    # fig, ax = plt.subplots()
    # ax.hist(predicted_clv_values, bins=20, color='skyblue', edgecolor='black')
    # ax.set_title("CLV Prediction Distribution")
    # ax.set_xlabel("CLV")
    # ax.set_ylabel("Frequency")
    # st.pyplot(fig)

if __name__ == "__main__":
    main()
