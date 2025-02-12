import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import time
import subprocess
from datetime import datetime
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

# Function to generate synthetic process data
def generate_synthetic_data(pid, setpoint, num_points):
    measurements = []
    control_signals = []
    measurement = np.random.normal(setpoint, 2)  # Initial measurement

    for _ in range(num_points):
        control_signal = pid.update(setpoint, measurement)
        measurement += control_signal + np.random.normal(0, 0.5)  # Simulate process response with noise
        measurements.append(measurement)
        control_signals.append(control_signal)

    return measurements, control_signals

# Load and prepare data
def load_data():
    data = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'feature3': np.random.rand(100),
        'failure': np.random.choice([0, 1], size=100)
    })
    X = data.drop('failure', axis=1)
    y = data['failure']
    return X, y

# Train model and get accuracy
def train_model(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return accuracy

# Sidebar
st.sidebar.title("Rocket Launch Control")
st.sidebar.header("Navigation")

# Navigation buttons
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

def set_page(page_name):
    st.session_state.page = page_name

st.sidebar.button("Home", on_click=set_page, args=('Home',))
st.sidebar.button("Synthetic Sensor Data & Model Training", on_click=set_page, args=('Synthetic Sensor Data & Model Training',))
st.sidebar.button("PID Controller Simulation", on_click=set_page, args=('PID Controller Simulation',))
st.sidebar.button("Rocket Environment", on_click=set_page, args=('Rocket Environment',))
# st.sidebar.button("Launch Sequencer", on_click=set_page, args=('Launch Sequencer',))
st.sidebar.button("Telemetry", on_click=set_page, args=('Telemetry',))
st.sidebar.button("Rocket Controller", on_click=set_page, args=('Rocket Controller',))

# Display selected page
if st.session_state.page == "Home":
    st.title("Welcome to the Rocket Launch Control System")
    
    # Banner Image
    st.image("isro _image.png", use_column_width=True)
    
    st.write("""
        ### Latest Rocket Launch Feed
        Stay updated with the latest rocket launches and events.
    """)
    
    # Placeholder for dynamic content
    st.subheader("Upcoming Launches")
    st.write("Launch Vehicle: HLVM3")
    st.write("Launch Date: 2025")
    st.write("Mission: Gaganyaan-Human rated LVM3 - HLVM3")
    st.write("Launch Site: ISRO INDIA")

    # Adding a horizontal line for better separation
    st.markdown("---")
    st.image("gangaayaan.png", use_column_width=True)

    st.subheader("Recent News")
    st.write("""
        - 2025: Gaganyaan-Human rated LVM3 - HLVM3
        - 2024: PSLV-C57/Aditya-L1 Mission
        - 2025: NASA-ISRO SAR (NISAR) Satellite
    """)
    # Adding a horizontal line for better separation
    st.markdown("---")
    
    # Add a call to action or contact information
    st.subheader("Contact Us")
    st.write("""
        For more information or inquiries, please reach out to us:
        - Email: info@rocketlaunch.com
        - Phone: +1-800-ROCKET
    """)

elif st.session_state.page == "Synthetic Sensor Data & Model Training":
    st.header("Synthetic Sensor Data & Model Training")

    with st.expander("View Sensor Data"):
        df = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'feature3': np.random.rand(100),
            'failure': np.random.choice([0, 1], size=100)
        })
        st.dataframe(df.head(), height=300)
        
    with st.expander("Train Model"):
        X, y = load_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        accuracy = train_model(X_train, y_train, X_test, y_test)
        st.metric(label="Model Accuracy", value=f"{accuracy * 100:.2f} %")

elif st.session_state.page == "PID Controller Simulation":
    st.header("PID Controller Simulation")
    with st.expander("Run Simulation"):
        kp = st.slider("Kp", 0.0, 5.0, 1.0, format="%.2f")
        ki = st.slider("Ki", 0.0, 5.0, 0.1, format="%.2f")
        kd = st.slider("Kd", 0.0, 5.0, 0.05, format="%.2f")
        setpoint = st.number_input("Setpoint", value=100)
        num_points = st.number_input("Number of data points", value=50)
        
        pid = PIDController(kp, ki, kd)
        measurements, control_signals = generate_synthetic_data(pid, setpoint, num_points)
        
        st.write("Synthetic Data Generated:")
        df = pd.DataFrame({
            "Measurement": measurements,
            "Control Signal": control_signals
        })
        st.line_chart(df)

        st.write("Control signal at the last step:", control_signals[-1])

elif st.session_state.page == "Rocket Environment":
    st.header("Rocket Environment")
    with st.expander("Get Environment Data"):
        try:
            env = Environment(latitude=32.990254, longitude=-106.974998, elevation=1400)
            valid_date = datetime.now()
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

elif st.session_state.page == "Telemetry":
    st.write("You will be redirected to the Telemetry page.")
    # Execute the telemetry.py file
    try:
        command = ['streamlit', 'run', 'C:\\Users\\pavan\\OneDrive\\Desktop\\AI-Rocketlaunch\\analysis\\telemetry.py']
        result = subprocess.Popen(command)
        st.write("Telemetry page is now running.")
    except Exception as e:
        st.error(f"An error occurred while trying to run telemetry.py: {e}")

elif st.session_state.page == "Rocket Controller":
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
