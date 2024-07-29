import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import time
import socket
from rocketpy import Environment
import subprocess
import os

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
        self._execute_streamlit_file('prelaunch_simulation/prelaunch_simulation_combined.py')
        self.pre_launch_checks_done = True

    def start_fueling(self):
        self._execute_streamlit_file('prefuelling/Prelaunchfueling.py')
        self.fueling_done = True

    def position_rocket(self):
        self._execute_streamlit_file('rocket_positioning/Rocket_positioning.py')
        self.positioning_done = True

    def launch(self):
        if self.pre_launch_checks_done and self.fueling_done and self.positioning_done:
            self._execute_streamlit_file('simulation_dash/Simulation Dashboard.py')
            sequencer = LaunchSequencer()
            sequencer.start_countdown()
        else:
            st.write("Cannot launch, not all systems are ready. Please complete all the steps.")

    def _execute_streamlit_file(self, file_name):
        try:
            command = ['streamlit', 'run', file_name]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0:
                st.write(f"{file_name} executed successfully.")
                st.write(result.stdout)
            else:
                st.write(f"Error executing {file_name}.")
                st.write(result.stderr)
        except FileNotFoundError:
            st.write(f"{file_name} not found.")
        except Exception as e:
            st.write(f"Unexpected error: {e}")

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
        s.connect(('ground_control_ip', 12345))  # Ensure 'ground_control_ip' is replaced with the actual IP
        s.sendall(data.encode())
        s.close()
        st.write("Telemetry data sent!")
    except Exception as e:
        st.write(f"Failed to send telemetry data: {e}")

# Streamlit app
st.title("Rocket Launch Control System")

# Synthetic Data
st.header("Synthetic Sensor Data")
try:
    df = pd.read_csv('synthetic_sensor_data.csv')
    st.write(df.head())
except FileNotFoundError:
    st.write("Synthetic sensor data file not found.")

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
try:
    env = Environment(latitude=32.990254, longitude=-106.974998, elevation=1400)
    valid_date = (2024, 7, 29, 6)  # Example: Adjust to a valid date within the range
    env.set_date(valid_date)
    env.set_atmospheric_model(type='Forecast', file='GFS')

    wind_speed = np.random.uniform(0, 20)  # Random wind speed between 0 and 20 m/s
    wind_direction = np.random.uniform(0, 360)  # Random wind direction between 0 and 360 degrees
    st.write(f"Wind speed: {wind_speed:.2f} m/s, Wind direction: {wind_direction:.2f} degrees")
except ValueError as e:
    st.write(f"Error: {e}")
except Exception as e:
    st.write(f"Error retrieving wind information: {e}")

# Launch Sequencer
st.header("Launch Sequencer")
countdown_time = st.slider("Countdown Time (seconds)", 0, 60, 10)
if st.button("Start Countdown"):
    sequencer = LaunchSequencer()
    sequencer.countdown = countdown_time
    sequencer.start_countdown()

# Telemetry
st.header("Send Telemetry Data")
telemetry_data = st.text_input("Telemetry Data", "altitude=1000;speed=5400")
if st.button("Send Telemetry"):
    send_telemetry(telemetry_data) 

# Rocket Controller section
st.header("Rocket Controller")
controller = RocketController()
if st.button("Run Pre-Launch Checks"):
    controller.run_pre_launch_checks()
    if controller.pre_launch_checks_done:
        st.write("Pre-launch checks completed.")
    else:
        st.write("Pre-launch checks failed.")
if st.button("Start Fueling"):
    controller.start_fueling()
    if controller.fueling_done:
        st.write("Fueling completed.")
    else:
        st.write("Fueling failed.")
if st.button("Position Rocket"):
    controller.position_rocket()
    if controller.positioning_done:
        st.write("Rocket positioned.")
    else:
        st.write("Rocket positioning failed.")
if st.button("Launch"):
    controller.launch()
