import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Constants
GRAVITY = 9.8  # m/s^2

# Functions to calculate parameters
def calculate_acceleration(t, mass):
    """Calculate acceleration at time t."""
    remaining_mass = mass - mass_flow_rate * t
    if remaining_mass <= 0:
        return -GRAVITY
    return thrust_force / remaining_mass - GRAVITY

def calculate_velocity(t, mass):
    """Calculate velocity at time t."""
    remaining_mass = mass - mass_flow_rate * t
    if remaining_mass <= 0:
        return max(-GRAVITY * t, 0)  # Ensure velocity is non-negative
    return max((thrust_force / mass_flow_rate) * np.log(mass / remaining_mass) - GRAVITY * t, 0)

def calculate_altitude(t, mass):
    """Calculate altitude at time t."""
    remaining_mass = mass - mass_flow_rate * t
    if remaining_mass <= 0:
        return -0.5 * GRAVITY * t**2
    return (thrust_force / mass_flow_rate) * ((mass / mass_flow_rate) * np.log(mass / remaining_mass) - t) - 0.5 * GRAVITY * t**2

def calculate_fuel_consumption(t):
    """Calculate remaining fuel mass at time t."""
    return initial_mass - mass_flow_rate * t

# Streamlit interface
st.title("Rocket Telemetry System")

# Sidebar for inputs
st.sidebar.header("Rocket Parameters")
initial_mass = st.sidebar.slider("Initial Mass (kg)", 100000, 1000000, 549054)
thrust_force = st.sidebar.slider("Thrust Force (N)", 100000, 2000000, 934000)
mass_flow_rate = st.sidebar.slider("Mass Flow Rate (kg/s)", 500, 5000, 2500)

# Input for time duration
st.sidebar.header("Simulation Settings")
time_duration = st.sidebar.slider("Select Time Duration (s)", 0, 162, 60)

# Check if mass becomes non-positive
mass_at_duration = initial_mass - mass_flow_rate * time_duration
if mass_at_duration <= 0:
    st.warning("Warning: Mass becomes non-positive within the selected time duration.")

# Calculate telemetry data
time_points = np.linspace(0, time_duration, num=500)
acceleration_data = [calculate_acceleration(t, initial_mass) for t in time_points]
velocity_data = [calculate_velocity(t, initial_mass) for t in time_points]
altitude_data = [calculate_altitude(t, initial_mass) for t in time_points]
fuel_data = [calculate_fuel_consumption(t) for t in time_points]

# Create a DataFrame for plotting
data = pd.DataFrame({
    "Time (s)": time_points,
    "Acceleration (m/s^2)": acceleration_data,
    "Velocity (m/s)": velocity_data,
    "Altitude (m)": altitude_data,
    "Fuel Mass (kg)": fuel_data
})

# Interactive Plotly Charts
def plot_interactive_line_chart(data, y_column, title):
    """Create an interactive Plotly line chart with hover information."""
    fig = px.line(data, x="Time (s)", y=y_column, title=title, markers=True)
    fig.update_traces(mode='lines+markers', hovertemplate=f'{y_column}: %{{y}}<br>Time: %{{x}}<extra></extra>')
    fig.update_layout(
        xaxis_title="Time (s)",
        yaxis_title=y_column,
        title=dict(x=0.5),
        template="plotly_white"
    )
    return fig

def plot_statistical_visualizations(data):
    """Create statistical visualizations such as histograms and scatter plots."""
    fig_hist_accel = px.histogram(data, x="Acceleration (m/s^2)", nbins=30, title="Histogram of Acceleration")
    fig_hist_vel = px.histogram(data, x="Velocity (m/s)", nbins=30, title="Histogram of Velocity")
    fig_hist_alt = px.histogram(data, x="Altitude (m)", nbins=30, title="Histogram of Altitude")
    
    fig_scatter = px.scatter(data, x="Velocity (m/s)", y="Altitude (m)", color="Fuel Mass (kg)",
                            title="Velocity vs. Altitude", color_continuous_scale=px.colors.sequential.Plasma)

    return fig_hist_accel, fig_hist_vel, fig_hist_alt, fig_scatter

# Display interactive charts
st.subheader("Telemetry Data Visualization")

# Creating figures
fig_acceleration = plot_interactive_line_chart(data, "Acceleration (m/s^2)", "Acceleration over Time")
fig_velocity = plot_interactive_line_chart(data, "Velocity (m/s)", "Velocity over Time")
fig_altitude = plot_interactive_line_chart(data, "Altitude (m)", "Altitude over Time")
fig_fuel = plot_interactive_line_chart(data, "Fuel Mass (kg)", "Fuel Mass over Time")

# Displaying two plots per row
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig_acceleration)
    st.plotly_chart(fig_altitude)
with col2:
    st.plotly_chart(fig_velocity)
    st.plotly_chart(fig_fuel)

# Display statistical visualizations
st.subheader("Statistical Visualizations")
fig_hist_accel, fig_hist_vel, fig_hist_alt, fig_scatter = plot_statistical_visualizations(data)

# Displaying two plots per row
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig_hist_accel)
    st.plotly_chart(fig_hist_alt)
with col2:
    st.plotly_chart(fig_hist_vel)
    st.plotly_chart(fig_scatter)

# Display the data table
st.subheader("Telemetry Data Table")
st.dataframe(data)

# Additional information and accessibility
st.sidebar.markdown("""
### Additional Information
- *Acceleration*: The rate of change of velocity.
- *Velocity*: The speed and direction of the rocket.
- *Altitude*: The height of the rocket above the Earth's surface.
- *Fuel Mass*: The remaining fuel mass over time.

### Accessibility
- Ensure that color contrast is sufficient for readability.
- Use screen readers or text-to-speech tools to access the information presented.
""")