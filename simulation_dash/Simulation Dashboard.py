import streamlit as st
import plotly.graph_objs as go
import numpy as np

# Constants
g0 = 9.7911  
g80 = 9.5534  
surface_elevation = 1471.5  

time = np.linspace(0, 100, 500)

def gravity(h):
    return g0 + (g80 - g0) * (h / 80111)

def rocket_trajectory(t):
    h = surface_elevation + 0.5 * g0 * t**2
    return h

altitude = rocket_trajectory(time)

# Launch Site Details
launch_date = "2024-07-27 06:00:00 UTC"
launch_latitude = "32.99025°"
launch_longitude = "-106.97500°"
launch_utm_coordinates = "315468.64 W    3651938.65 N"
launch_utm_zone = "13S"
launch_surface_elevation = "1471.5 m"

# Atmospheric Model Details
atmospheric_model_type = "Forecast"
forecast_max_height = "80.111 km"
forecast_time_period = "from 2024-07-27 06:00:00 to 2024-08-12 06:00:00 UTC"
forecast_hour_interval = "3 hrs"
forecast_latitude_range = "From -90.0° to 90.0°"
forecast_longitude_range = "From 0.0° to 359.75°"

# Surface Atmospheric Conditions
surface_wind_speed = "354.40 m/s"
surface_wind_direction = "174.40°"
surface_wind_heading = "5.18°"
surface_pressure = "853.33 hPa"
surface_temperature = "299.95 K"
surface_air_density = "0.991 kg/m³"
surface_speed_of_sound = "347.19 m/s"

# Earth Model Details
earth_radius = "6371.83 km"
semi_major_axis = "6378.14 km"
semi_minor_axis = "6356.75 km"
flattening = "0.0034"

st.title("Rocket Launch Simulation Dashboard")

# Rocket Trajectory Plot
fig = go.Figure()

# Add a scatter plot for the trajectory
fig.add_trace(go.Scatter(
    x=time,
    y=altitude,
    mode='lines',
    line=dict(color='blue'),
    name='Trajectory'
))

# Create animation frames
frames = [go.Frame(
    data=[go.Scatter(
        x=time[:k],
        y=rocket_trajectory(time[:k]),
        mode='lines',
        line=dict(color='blue')
    )],
    name=f'Frame {k}'
) for k in range(1, len(time) + 1)]

fig.frames = frames

fig.update_layout(
    xaxis=dict(range=[0, 100], title='Time (s)'),
    yaxis=dict(range=[0, max(altitude)+1000], title='Altitude (m)'),
    title='Rocket Launch Simulation',
    updatemenus=[{
        'type': 'buttons',
        'buttons': [{
            'label': 'Play',
            'method': 'animate',
            'args': [None, {'frame': {'duration': 50, 'redraw': True}, 'fromcurrent': True}]
        }]
    }]
)

st.plotly_chart(fig)

# Display Launch Site Details
st.subheader("Launch Site Details")
st.write(f"Launch Date: {launch_date}")
st.write(f"Launch Site Latitude: {launch_latitude}")
st.write(f"Launch Site Longitude: {launch_longitude}")
st.write(f"Launch Site UTM coordinates: {launch_utm_coordinates}")
st.write(f"Launch Site UTM zone: {launch_utm_zone}")
st.write(f"Launch Site Surface Elevation: {launch_surface_elevation}")

# Display Atmospheric Model Details
st.subheader("Atmospheric Model Details")
st.write(f"Atmospheric Model Type: {atmospheric_model_type}")
st.write(f"Forecast Maximum Height: {forecast_max_height}")
st.write(f"Forecast Time Period: {forecast_time_period}")
st.write(f"Forecast Hour Interval: {forecast_hour_interval}")
st.write(f"Forecast Latitude Range: {forecast_latitude_range}")
st.write(f"Forecast Longitude Range: {forecast_longitude_range}")

# Display Surface Atmospheric Conditions
st.subheader("Surface Atmospheric Conditions")
st.write(f"Surface Wind Speed: {surface_wind_speed}")
st.write(f"Surface Wind Direction: {surface_wind_direction}")
st.write(f"Surface Wind Heading: {surface_wind_heading}")
st.write(f"Surface Pressure: {surface_pressure}")
st.write(f"Surface Temperature: {surface_temperature}")
st.write(f"Surface Air Density: {surface_air_density}")
st.write(f"Surface Speed of Sound: {surface_speed_of_sound}")

# Display Earth Model Details
st.subheader("Earth Model Details")
st.write(f"Earth Radius at Launch site: {earth_radius}")
st.write(f"Semi-major Axis: {semi_major_axis}")
st.write(f"Semi-minor Axis: {semi_minor_axis}")
st.write(f"Flattening: {flattening}")
