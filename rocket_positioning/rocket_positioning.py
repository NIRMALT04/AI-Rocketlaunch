import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Function to generate synthetic environmental data
def generate_synthetic_data(steps, wind_speed_range, wind_direction_range):
    timestamps = np.arange(steps)
    wind_speeds = np.random.uniform(*wind_speed_range, size=steps)
    wind_directions = np.random.uniform(*wind_direction_range, size=steps)
    return timestamps, wind_speeds, wind_directions

# Function to calculate wind force on the rocket
def calculate_wind_force(wind_speed, wind_direction, rocket_orientation):
    drag_coefficient = 1.0  # Assumed drag coefficient for a cylindrical object
    wind_force = 0.5 * air_density * cross_sectional_area * drag_coefficient * wind_speed**2
    force_direction = wind_direction - rocket_orientation
    return wind_force * np.cos(np.deg2rad(force_direction)), wind_force * np.sin(np.deg2rad(force_direction))

# Title
st.title('Rocket Positioning Simulation')

# Sidebar inputs
st.sidebar.header('Simulation Parameters')
rocket_mass = st.sidebar.number_input('Rocket Mass (kg)', value=50000)
rocket_height = st.sidebar.number_input('Rocket Height (meters)', value=50)
rocket_diameter = st.sidebar.number_input('Rocket Diameter (meters)', value=5)
simulation_steps = st.sidebar.number_input('Number of Simulation Steps', value=100, min_value=10, max_value=1000)
wind_speed_range = st.sidebar.slider('Wind Speed Range (m/s)', 0.0, 20.0, (0.0, 10.0))
wind_direction_range = st.sidebar.slider('Wind Direction Range (degrees)', 0.0, 360.0, (0.0, 360.0))

# Generate synthetic data
timestamps, wind_speeds, wind_directions = generate_synthetic_data(simulation_steps, wind_speed_range, wind_direction_range)

# Constants
cross_sectional_area = np.pi * (rocket_diameter / 2) ** 2
air_density = 1.225  # kg/m^3 (at sea level)

# Initial rocket position and orientation
rocket_position = np.array([0.0, 0.0])  # [x, y] in meters
rocket_orientation = 0.0  # In degrees

# Store rocket positions over time
positions = [rocket_position]

# Simulate rocket positioning over time
for wind_speed, wind_direction in zip(wind_speeds, wind_directions):
    wind_force_x, wind_force_y = calculate_wind_force(wind_speed, wind_direction, rocket_orientation)
    rocket_position += np.array([wind_force_x, wind_force_y]) / rocket_mass
    positions.append(rocket_position)

# Convert positions to numpy array for plotting
positions = np.array(positions)

# Plot rocket position over time
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(positions[:, 0], positions[:, 1], marker='o')
ax.set_title('Rocket Position on Launch Pad Over Time')
ax.set_xlabel('X Position (meters)')
ax.set_ylabel('Y Position (meters)')
ax.grid()

st.pyplot(fig)

# Show data
st.subheader('Environmental Data')
data = pd.DataFrame({
    'timestamp': timestamps,
    'wind_speed': wind_speeds,
    'wind_direction': wind_directions
})
st.write(data)
