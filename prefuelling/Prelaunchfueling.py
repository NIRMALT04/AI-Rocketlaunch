import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import seaborn as sns

# Data Preprocessing Function
def preprocess_data(data):
    data.fillna(data.mean(), inplace=True)
    return data

# Simulated Pre-launch Checks
def check_systems(data):
    return {"propulsion": True, "navigation": True, "communication": True, "structural_integrity": True}

def analyze_sensor_data(sensor_data):
    return sensor_data.mean(axis=0)

# Training Fuel Models
def train_fuel_models(fuel_data, fuel_labels):
    lin_reg = Pipeline([('scaler', StandardScaler()), ('regressor', LinearRegression())])
    lin_reg.fit(fuel_data, fuel_labels)
    
    rf_reg = Pipeline([('scaler', StandardScaler()), ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])
    rf_reg.fit(fuel_data, fuel_labels)
    
    svr_reg = Pipeline([('scaler', StandardScaler()), ('regressor', SVR(kernel='rbf'))])
    svr_reg.fit(fuel_data, fuel_labels)
    
    return lin_reg, rf_reg, svr_reg

# Predicting Fuel Requirements
def predict_fuel_requirement(models, fuel_data):
    lin_reg, rf_reg, svr_reg = models
    lin_reg_pred = lin_reg.predict(fuel_data)
    rf_reg_pred = rf_reg.predict(fuel_data)
    svr_reg_pred = svr_reg.predict(fuel_data)
    final_prediction = (lin_reg_pred + rf_reg_pred + svr_reg_pred) / 3
    return lin_reg_pred, rf_reg_pred, svr_reg_pred, final_prediction

# Safety Protocols
def safety_protocols(fuel_level, max_capacity, min_capacity=0):
    if fuel_level > max_capacity:
        return "Error: Overfueling detected. Abort fueling process."
    elif fuel_level < min_capacity:
        return "Error: Underfueling detected. Check fuel system."
    else:
        return "Fuel level within safe range. Proceed."

# Simulated Data for Demonstration
fuel_data = pd.DataFrame({
    'mission_duration': np.random.uniform(1, 10, 100),
    'payload_mass': np.random.uniform(500, 2000, 100),
    'destination_distance': np.random.uniform(100, 1000, 100)
})
fuel_labels = np.random.uniform(1000, 5000, 100)
sensor_data = pd.DataFrame({
    'temperature': np.random.uniform(-50, 50, 100),
    'pressure': np.random.uniform(0.5, 1.5, 100),
    'vibration': np.random.uniform(0, 5, 100),
    'humidity': np.random.uniform(0, 100, 100)
})
example_fuel_data = pd.DataFrame({
    'mission_duration': [5],
    'payload_mass': [1200],
    'destination_distance': [500]
})
max_fuel_capacity = 4000

# Preprocess Data
fuel_data = preprocess_data(fuel_data)

# Pre-launch Checks
system_status = check_systems(sensor_data)
analyzed_sensor_data = analyze_sensor_data(sensor_data)

# Train Models for Fuel Calculations
models = train_fuel_models(fuel_data, fuel_labels)

# Predict Fuel Requirement
lin_reg_pred, rf_reg_pred, svr_reg_pred, predicted_fuel = predict_fuel_requirement(models, example_fuel_data)

# Safety Protocols Check
current_fuel_level = predicted_fuel[0]
safety_status = safety_protocols(current_fuel_level, max_fuel_capacity)

# Full Pipeline for Autonomous Launch Preparation Process
def autonomous_launch_preparation(fuel_data, fuel_labels, sensor_data, example_fuel_data, max_fuel_capacity):
    fuel_data = preprocess_data(fuel_data)
    system_status = check_systems(sensor_data)
    analyzed_sensor_data = analyze_sensor_data(sensor_data)
    models = train_fuel_models(fuel_data, fuel_labels)
    lin_reg_pred, rf_reg_pred, svr_reg_pred, predicted_fuel = predict_fuel_requirement(models, example_fuel_data)
    current_fuel_level = predicted_fuel[0]
    safety_status = safety_protocols(current_fuel_level, max_fuel_capacity)
    return system_status, analyzed_sensor_data, lin_reg_pred, rf_reg_pred, svr_reg_pred, predicted_fuel, safety_status

# Running the Autonomous Launch Preparation Process
system_status, analyzed_sensor_data, lin_reg_pred, rf_reg_pred, svr_reg_pred, predicted_fuel, safety_status = autonomous_launch_preparation(
    fuel_data, fuel_labels, sensor_data, example_fuel_data, max_fuel_capacity)

# Streamlit Visualization
st.title('Autonomous Launch Preparation System')

# Sensor Data Visualization
st.subheader('Analyzed Sensor Data')
fig, ax = plt.subplots()
sns.barplot(x=analyzed_sensor_data.index, y=analyzed_sensor_data.values, ax=ax)
ax.set_title('Analyzed Sensor Data')
ax.set_ylabel('Value')
ax.set_xlabel('Sensor Type')
st.pyplot(fig)

# Fuel Prediction Visualization
st.subheader('Predicted Fuel Requirement')
fig, ax = plt.subplots()
sns.barplot(x=['Linear Regression', 'Random Forest', 'SVR', 'Averaged Prediction'], y=[lin_reg_pred[0], rf_reg_pred[0], svr_reg_pred[0], predicted_fuel[0]], ax=ax)
ax.set_title('Predicted Fuel Requirement')
ax.set_ylabel('Fuel (liters)')
ax.set_xlabel('Model')
st.pyplot(fig)

# Mission Duration Distribution
st.subheader('Mission Duration Distribution')
fig, ax = plt.subplots()
sns.histplot(fuel_data['mission_duration'], kde=True, ax=ax)
ax.set_title('Mission Duration Distribution')
ax.set_xlabel('Mission Duration (hours)')
st.pyplot(fig)

# Payload Mass Distribution
st.subheader('Payload Mass Distribution')
fig, ax = plt.subplots()
sns.histplot(fuel_data['payload_mass'], kde=True, ax=ax)
ax.set_title('Payload Mass Distribution')
ax.set_xlabel('Payload Mass (kg)')
st.pyplot(fig)

# Display Safety Status
st.subheader('Safety Protocols')
st.write(safety_status)
