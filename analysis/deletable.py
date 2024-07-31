import streamlit as st
import pandas as pd
import numpy as np
import time
from collections import deque

# Constants
G = 9.81  # gravity constant in m/s^2
EARTH_RADIUS = 6371000  # in meters
ESCAPE_VELOCITY = 11186  # in m/s

# Initialize data storage
data_queue = deque(maxlen=10)

# Function to generate telemetry data
def generate_telemetry_data():
    initial_mass = 500000  # kg
    dry_mass = 200000      # kg
    fuel_mass = initial_mass - dry_mass
    altitude = 0           # meters
    velocity = 0           # meters per second
    acceleration = 0       # meters per second squared
    temperature = 15       # degrees Celsius
    pressure = 101.3       # kPa

    while fuel_mass > 0 and altitude < EARTH_RADIUS + ESCAPE_VELOCITY:
        fuel_consumption_rate = np.random.uniform(50, 150)  # kg/s
        thrust = np.random.uniform(1e6, 1.5e6)  # Newtons

        # Update values
        acceleration = (thrust / (dry_mass + fuel_mass)) - G
        velocity += acceleration * 0.1  # 0.1 second interval for faster updates
        altitude += velocity * 0.1  # 0.1 second interval for faster updates
        fuel_mass -= fuel_consumption_rate * 0.1

        if altitude >= EARTH_RADIUS:
            gravity = G * (EARTH_RADIUS / altitude) ** 2
        else:
            gravity = G
        
        temperature = np.random.uniform(-50, 50)
        pressure = 101.3 * np.exp(-altitude / 8500)  # Barometric formula for pressure

        data = {
            "initial_mass": initial_mass,
            "dry_mass": dry_mass,
            "fuel_mass": fuel_mass,
            "altitude": altitude,
            "velocity": velocity,
            "acceleration": acceleration,
            "fuel_consumption_rate": fuel_consumption_rate,
            "temperature": temperature,
            "pressure": pressure
        }

        yield data
        time.sleep(0.1)  # faster update interval

# Set up Streamlit layout
st.title("Rocket Telemetry Monitoring System")

# Real-time data display
st.subheader("Real-Time Telemetry Data")

# Create containers for real-time data display
altitude_container = st.empty()
velocity_container = st.empty()
acceleration_container = st.empty()
fuel_mass_container = st.empty()
fuel_consumption_rate_container = st.empty()
temperature_container = st.empty()
pressure_container = st.empty()

# Historical data display
st.subheader("Historical Telemetry Data")
historical_data_chart = st.line_chart(pd.DataFrame(columns=["altitude", "velocity", "acceleration", "fuel_mass", "fuel_consumption_rate", "temperature", "pressure"]))

# Simulate receiving data in real-time
telemetry_data_generator = generate_telemetry_data()
for telemetry_data in telemetry_data_generator:
    data_queue.append(telemetry_data)

    # Display real-time data
    altitude_container.metric(label="Altitude (m)", value=f"{telemetry_data['altitude']:.2f}")
    velocity_container.metric(label="Velocity (m/s)", value=f"{telemetry_data['velocity']:.2f}")
    acceleration_container.metric(label="Acceleration (m/s²)", value=f"{telemetry_data['acceleration']:.2f}")
    fuel_mass_container.metric(label="Fuel Mass (kg)", value=f"{telemetry_data['fuel_mass']:.2f}")
    fuel_consumption_rate_container.metric(label="Fuel Consumption Rate (kg/s)", value=f"{telemetry_data['fuel_consumption_rate']:.2f}")
    temperature_container.metric(label="Temperature (°C)", value=f"{telemetry_data['temperature']:.2f}")
    pressure_container.metric(label="Pressure (kPa)", value=f"{telemetry_data['pressure']:.2f}")

    # Update historical data chart
    historical_data_chart.add_rows(pd.DataFrame(data_queue))
