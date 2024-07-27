import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import time
import socket
from rocketpy import Environment

# Define your classes here

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.previous_error = 0

    def update(self, setpoint, measurement):
        error = setpoint - measurement
        self.integral += error
        derivative = error - self.previous_error
        self.previous_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

class LaunchSequencer:
    def __init__(self):
        self.countdown = 10

    def start_countdown(self):
        while self.countdown > 0:
            st.write(f"T-minus {self.countdown} seconds")
            time.sleep(1)
            self.countdown -= 1
        st.write("Launch!")

class RocketController:
    def __init__(self):
        self.pre_launch_checks_done = False
        self.fueling_done = False
        self.positioning_done = False

    def run_pre_launch_checks(self):
        self.pre_launch_checks_done = True

    def start_fueling(self):
        self.fueling_done = True

    def position_rocket(self):
        self.positioning_done = True

    def launch(self):
        if self.pre_launch_checks_done and self.fueling_done and self.positioning_done:
            sequencer = LaunchSequencer()
            sequencer.start_countdown()
        else:
            st.write("Cannot launch, not all systems are ready")

# Load and prepare data
def load_data():
    data = pd.read_csv('synthetic_sensor_data.csv')
    X = data.drop('failure', axis=1)
    y = data['failure']
    return X, y

# Train model and get accuracy
def train_model(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return accuracy

# Send telemetry data
def send_telemetry(data):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('ground_control_ip', 12345))
        s.sendall(data.encode())
        s.close()
        st.write("Telemetry data sent!")
    except Exception as e:
        st.write(f"Failed to send telemetry data: {e}")

# Streamlit app
st.title("Rocket Launch Control System")

# Synthetic Data
st.header("Synthetic Sensor Data")
df = pd.read_csv('synthetic_sensor_data.csv')
st.write(df.head())

# Model Training
st.header("Model Training")
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
accuracy = train_model(X_train, y_train, X_test, y_test)
st.write(f"Model accuracy: {accuracy * 100:.2f}%")

# PID Controller Simulation
st.header("PID Controller Simulation")
kp = st.slider("Kp", 0.0, 5.0, 1.0)
ki = st.slider("Ki", 0.0, 5.0, 0.1)
kd = st.slider("Kd", 0.0, 5.0, 0.05)
setpoint = 100
measurement = 80
pid = PIDController(kp, ki, kd)
control_signal = pid.update(setpoint, measurement)
st.write(f"Control signal: {control_signal}")

# Rocket Environment
st.header("Rocket Environment")
Env = Environment(latitude=32.990254, longitude=-106.974998, elevation=1400)
Env.set_date((2024, 7, 27, 6))  
Env.set_atmospheric_model(type='Forecast', file='GFS')

# Placeholder for wind information
try:
    # Replace these with the correct attributes once you verify the API
    wind_speed = Env.get_wind_speed()  # Placeholder method
    wind_direction = Env.get_wind_direction()  # Placeholder method
    st.write(f"Wind speed: {wind_speed} m/s, Wind direction: {wind_direction} degrees")
except AttributeError:
    st.write("Wind information not available. Check the `rocketpy` documentation for the latest API changes.")

# Launch Sequencer
st.header("Launch Sequencer")
countdown_time = st.slider("Countdown Time (seconds)", 0, 60, 10)
if st.button("Start Countdown"):
    sequencer = LaunchSequencer()
    sequencer.countdown = countdown_time
    for i in range(countdown_time, 0, -1):
        st.write(f"T-minus {i} seconds")
        time.sleep(1)
    st.write("Launch!")

# Telemetry
st.header("Send Telemetry Data")
telemetry_data = st.text_input("Telemetry Data", "altitude=1000;speed=5400")
if st.button("Send Telemetry"):
    send_telemetry(telemetry_data) 

# Rocket Controller
st.header("Rocket Controller")
controller = RocketController()
if st.button("Run Pre-Launch Checks"):
    controller.run_pre_launch_checks()
    st.write("Pre-launch checks completed.")
if st.button("Start Fueling"):
    controller.start_fueling()
    st.write("Fueling completed.")
if st.button("Position Rocket"):
    controller.position_rocket()
    st.write("Rocket positioned.")
if st.button("Launch"):
    controller.launch()
