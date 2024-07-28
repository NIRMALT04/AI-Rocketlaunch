import streamlit as st
import plotly.graph_objs as go
import numpy as np
import time as time_lib
from datetime import datetime

# Constants
g0 = 9.7911
g80 = 9.5534
surface_elevation = 1471.5

time = np.linspace(0, 100, 500)
fuel_capacity = 1000  # in kg
fuel_remaining = fuel_capacity

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

# Create empty placeholders
trajectory_placeholder = st.empty()
fuel_placeholder = st.empty()
launch_details_placeholder = st.empty()

# Function to update the plot
def update_plot():
    fig = go.Figure()
    for t in time:
        fig.add_trace(go.Scatter(
            x=[t],
            y=[rocket_trajectory(t)],
            mode='markers',
            marker=dict(
                size=10,
                symbol='triangle-up',
                color='blue'
            ),
            name=str(t)
        ))

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
        }],
        annotations=[
            dict(
                x=0.5,
                y=0.95,
                xref='paper',
                yref='paper',
                text=f'Temperature: {surface_temperature}<br>Pressure: {surface_pressure}<br>Fuel: {fuel_remaining}kg',
                showarrow=False,
                font=dict(size=14, color='black'),
                align='center'
            )
        ]
    )

    return fig

# Simulate real-time updates
start_time = time_lib.time()
while True:
    current_time = time_lib.time() - start_time

    # Update fuel remaining
    if fuel_remaining > 0:
        fuel_remaining -= 1  # Decrease fuel over time
    
    # Update plot
    fig = update_plot()
    trajectory_placeholder.plotly_chart(fig)

    # Update fuel status
    fuel_placeholder.subheader("Fuel Status")
    fuel_placeholder.write(f"Fuel Remaining: {fuel_remaining} kg")

    # Update Launch Site Details
    launch_details_placeholder.subheader("Launch Site Details")
    launch_details_placeholder.write(f"Launch Date: {launch_date}")
    launch_details_placeholder.write(f"Launch Site Latitude: {launch_latitude}")
    launch_details_placeholder.write(f"Launch Site Longitude: {launch_longitude}")
    launch_details_placeholder.write(f"Launch Site UTM coordinates: {launch_utm_coordinates}")
    launch_details_placeholder.write(f"Launch Site UTM zone: {launch_utm_zone}")
    launch_details_placeholder.write(f"Launch Site Surface Elevation: {launch_surface_elevation}")

    # Sleep for a while before next update
    time_lib.sleep(1)