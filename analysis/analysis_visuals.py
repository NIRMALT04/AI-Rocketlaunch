import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import time
import socket
from rocketpy import Environment
import subprocess

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
        st.success("Launch!")

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

    def launch_trajectory(self):
        self._execute_streamlit_file('simulation_dash/Simulation Dashboard.py')
        sequencer = LaunchSequencer()
        sequencer.start_countdown()

    def launch_simulation(self):
        self._execute_streamlit_file('3D_simulation.py')  # Adjust file name as needed

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

# def send_telemetry(data):
#     ground_control_ip = '127.0.0.1'
#     ground_control_port = 12345

#     try:
#         s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         s.settimeout(10)
#         s.connect((ground_control_ip, ground_control_port))
#         s.sendall(data.encode())
#         s.close()
#         st.success("Telemetry data sent!")
#     except ConnectionRefusedError:
#         st.error("Connection refused by the target machine. Ensure the service is running and listening on the port.")
#     except socket.timeout:
#         st.error("Connection timed out. The target machine may not be responding.")
#     except socket.gaierror as e:
#         st.error(f"Address-related error connecting to server: {e}")
#     except socket.error as e:
#         st.error(f"Socket error: {e}")
#     except Exception as e:
#         st.error(f"Failed to send telemetry data: {e}")

    #     st.error(f"Failed to send telemetry data: {e}")
# Sidebar
st.sidebar.title("Rocket Launch Control")
st.sidebar.header("Navigation")

# Navigation buttons
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

def set_page(page_name):
    st.session_state.page = page_name

st.sidebar.button("Home", on_click=set_page, args=('Home',), key='home_button_1')
st.sidebar.button("Synthetic Sensor Data", on_click=set_page, args=('Synthetic Sensor Data',), key='data_button_1')
st.sidebar.button("Model Training", on_click=set_page, args=('Model Training',), key='training_button_1')
st.sidebar.button("PID Controller Simulation", on_click=set_page, args=('PID Controller Simulation',), key='pid_button_1')
st.sidebar.button("Rocket Environment", on_click=set_page, args=('Rocket Environment',), key='environment_button_1')
st.sidebar.button("Launch Sequencer", on_click=set_page, args=('Launch Sequencer',), key='sequencer_button_1')
# st.sidebar.button("Telemetry", on_click=set_page, args=('Telemetry',), key='telemetry_button_1')
st.sidebar.button("Rocket Controller", on_click=set_page, args=('Rocket Controller',), key='controller_button_1')

# Display selected page
if st.session_state.page == "Home":
    st.title("🚀 Welcome to the Rocket Launch Control System!")
    
    # Banner Image
    st.image("isro _image.png", use_column_width=True)
    
    st.write("""
        ### Latest Rocket Launch Feed
        Stay updated with the latest rocket launches and events.
    """)
    
    # Placeholder for dynamic content
    st.subheader("Upcoming Launches")
    st.write("**Launch Vehicle:** Falcon 9")
    st.write("**Launch Date:** August 5, 2024")
    st.write("**Mission:** Starlink 7")
    st.write("**Launch Site:** Cape Canaveral Space Force Station")

    # Adding a horizontal line for better separation
    st.markdown("---")

    st.subheader("Recent News")
    st.write("""
        - **July 28, 2024:** SpaceX successfully launched the Starship prototype.
        - **July 25, 2024:** NASA's Artemis I mission is in the final stages of preparation.
        - **July 20, 2024:** Blue Origin announces a new lunar lander design.
    """)

    # Adding a horizontal line for better separation
    st.markdown("---")
    
    # Add a call to action or contact information
    st.subheader("Contact Us")
    st.write("""
        For more information or inquiries, please reach out to us:
        - **Email:** info@rocketlaunch.com
        - **Phone:** +1-800-ROCKET
    """)

if st.session_state.page == "Synthetic Sensor Data":
    st.header("Synthetic Sensor Data")
    with st.expander("View Sensor Data"):
        try:
            df = pd.read_csv('synthetic_sensor_data.csv')
            st.dataframe(df.head(), height=300)
        except FileNotFoundError:
            st.error("Synthetic sensor data file not found.")

if st.session_state.page == "Model Training":
    st.header("Model Training")
    with st.expander("Train Model"):
        X, y = load_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        accuracy = train_model(X_train, y_train, X_test, y_test)
        st.metric(label="Model Accuracy", value=f"{accuracy * 100:.2f} %")

if st.session_state.page == "PID Controller Simulation":
    st.header("PID Controller Simulation")
    with st.expander("Run Simulation"):
        kp = st.slider("Kp", 0.0, 5.0, 1.0, format="%.2f")
        ki = st.slider("Ki", 0.0, 5.0, 0.1, format="%.2f")
        kd = st.slider("Kd", 0.0, 5.0, 0.05, format="%.2f")
        setpoint = 100
        measurement = 80
        pid = PIDController(kp, ki, kd)
        control_signal = pid.update(setpoint, measurement)
        st.write(f"Control signal: {control_signal:.2f}")

if st.session_state.page == "Rocket Environment":
    st.header("Rocket Environment")
    with st.expander("Get Environment Data"):
        try:
            env = Environment(latitude=32.990254, longitude=-106.974998, elevation=1400)
            valid_date = (2024, 7, 29, 6)
            env.set_date(valid_date)
            env.set_atmospheric_model(type='Forecast', file='GFS')

            wind_speed = np.random.uniform(0, 20)
            wind_direction = np.random.uniform(0, 360)
            st.metric(label="Wind Speed (m/s)", value=f"{wind_speed:.2f}")
            st.metric(label="Wind Direction (degrees)", value=f"{wind_direction:.2f}")
        except ValueError as e:
            st.error(f"Error: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

if st.session_state.page == "Launch Sequencer":
    st.header("Launch Sequencer")
    if st.button("Start Countdown"):
        sequencer = LaunchSequencer()
        sequencer.start_countdown()

# if st.session_state.page == "Telemetry":
#     st.header("Send Telemetry Data")
#     with st.expander("Send Data"):
#         telemetry_data = st.text_input("Telemetry Data", "altitude=1000;speed=5400")
#         if st.button("Send Telemetry"):
#             send_telemetry(telemetry_data)

if st.session_state.page == "Rocket Controller":
    st.header("Rocket Controller")
    with st.expander("Control Rocket"):
        controller = RocketController()
        if st.button("Run Pre-launch Checks"):
            controller.run_pre_launch_checks()
        if st.button("Start Fueling"):
            controller.start_fueling()
        if st.button("Position Rocket"):
            controller.position_rocket()
        if st.button("Launch Trajectory"):
            controller.launch_trajectory()
        if st.button("Launch Simulation"):
            controller.launch_simulation()
