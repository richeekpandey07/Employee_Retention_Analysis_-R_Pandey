# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# 1. This MUST be the absolute first Streamlit command
st.set_page_config(
    page_title="Employee Retention Analysis",
    page_icon="🤝",
    layout="centered"
)

# Title & Info
st.title("Employee Retention Analysis")
st.write("Predicting employee turnover using Logistic Regression.")

st.markdown("---")
st.markdown("""
👨‍💻 **Developed By:** RICHEEK_PANDEY
📧 **Branch:** INFORMATION_TECHNOLOGY | **Room:** G-612
🔗 **LinkedIn:** [Profile](https://www.linkedin.com/in/richeek-pandey-9954783a9)
💻 **GitHub:** [Repository](https://github.com/richeekpandey07)
""")

# Load dataset safely
try:
    df = pd.read_csv("HR_comma_sep (2).csv")
except FileNotFoundError:
    st.error("❌ Error: 'HR_comma_sep (2).csv' not found. Please place the file in the same directory.")
    st.stop()

# --- DATA EXPLORATION ---
st.header("📊 Data Exploration")

if st.checkbox("Show Raw Data"):
    st.dataframe(df.head())

if st.checkbox("Show Mean Values Grouped by Retention"):
    st.dataframe(df.groupby('left').mean(numeric_only=True))

# --- VISUALIZATIONS ---
st.header("📈 Data Visualizations")

# Chart 1: Salary vs Retention
st.subheader("Salary vs Employee Retention")
fig1, ax1 = plt.subplots()
pd.crosstab(df.salary, df.left).plot(kind='bar', ax=ax1)
ax1.set_xlabel("Salary")
ax1.set_ylabel("Number of Employees")
st.pyplot(fig1)

# Chart 2: Department vs Retention
st.subheader("Department vs Employee Retention")
fig2, ax2 = plt.subplots(figsize=(10, 5))
pd.crosstab(df.Department, df.left).plot(kind='bar', ax=ax2)
ax2.set_xlabel("Department")
ax2.set_ylabel("Number of Employees")
st.pyplot(fig2)

# --- MODEL TRAINING ---
X = df[['satisfaction_level', 'average_montly_hours', 'promotion_last_5years', 'salary']].copy()

# Encode Salary
le = LabelEncoder()
X['salary'] = le.fit_transform(X['salary'])
y = df['left']

# Split & Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Calculate Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Sidebar Metrics
st.sidebar.header("🤖 Model Performance")
st.sidebar.metric(label="Model Accuracy", value=f"{accuracy:.2%}")

# --- INTERACTIVE PREDICTION ---
st.markdown("---")
st.header("🔮 Predict Employee Retention")

satisfaction = st.slider("Satisfaction Level", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
hours = st.number_input("Average Monthly Hours", min_value=10.0, max_value=400.0, value=200.0, step=1.0)
promotion = st.selectbox("Promoted in last 5 years?", options=["No", "Yes"])
salary_input = st.selectbox("Salary Level", options=list(le.classes_))

# Convert inputs for the model
promotion_val = 1 if promotion == "Yes" else 0
salary_val = le.transform([salary_input])[0]

if st.button("Predict"):
    input_data = pd.DataFrame([[satisfaction, hours, promotion_val, salary_val]], 
                              columns=['satisfaction_level', 'average_montly_hours', 'promotion_last_5years', 'salary'])
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    if prediction == 1:
        st.error(f"⚠️ Warning: The employee is likely to **LEAVE**. (Risk Probability: {probability:.2%})")
    else:
        st.success(f"✅ Success: The employee is likely to **STAY**. (Retention Probability: {(1-probability):.2%})")
