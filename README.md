# Customer Lifetime Value (CLV) Prediction with Machine Learning

## 📑 Overview

This project aims to predict **Customer Lifetime Value (CLV)** using machine learning models. CLV helps businesses forecast future revenue from customers, enabling data-driven decisions for marketing and customer relationship management.

The project leverages **RFM analysis** (Recency, Frequency, Monetary) for customer segmentation and predictive modeling. An interactive web app is developed using **Streamlit** for real-time CLV predictions.

---

## 🔍 Problem Statement

**Goal**: To predict the future value a customer brings to a business, based on past transactions. This enables businesses to:
- Segment customers based on their buying behavior.
- Identify high-value customers for targeted marketing.
- Forecast revenue contributions over time.

---

## 🛠️ Tech Stack

- **Python**
- **Pandas, NumPy**: Data manipulation and analysis
- **Feature Engineering**:Extracting important features
- **Scikit-learn, Machine learning algorithms
- **Streamlit**: Web app for real-time CLV predictions
- **Matplotlib, Seaborn**: Data visualization
- **Joblib**: Model serialization
- **VSCode**: Development environment

---

## 🚀 Solution

### 1. **RFM Analysis for Customer Segmentation**
   - **Recency**: How recently a customer made a purchase.
   - **Frequency**: How often a customer makes a purchase.
   - **Monetary**: The total value of a customer's purchases.
   
   Using **RFM** scores, customers are segmented into clusters using **K-Means clustering**, helping identify high-value customers and their behavioral patterns.

### 2. **Model Building**
   Various regression models were implemented to predict CLV, including:
   - **Linear Regression**
   - **Random Forest**
   - **AdaBoost**
   - **Gradient Boosting**
   - **XGBoost**

### 3. **Model Optimization**
   Hyperparameter tuning was performed using **RandomizedSearchCV** to improve model performance.

### 4. **Deployment**
   The predictive model is deployed through an interactive **Streamlit** app that allows users to input customer data and get real-time CLV predictions.

---

## 💻 How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repository-name.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run main.py
   ```

---

## 📂 Directory Structure

```
├── models                   # Trained models
├── data                     # Data files
├── notebooks                # Jupyter/Colab notebooks
├── main.py                  # Streamlit app
├── requirements.txt         # Dependencies
└── README.md                # Documentation
```

---

## 📚 Future Enhancements

- Incorporate more features such as customer demographics for better predictions.
- Explore time-series forecasting for CLV trends.
- Extend the web app by adding customer data visualizations.

---
