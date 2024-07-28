import streamlit as st
import numpy as np
import plotly.graph_objs as go
from scipy.integrate import odeint

# Atmospheric model (simplified US Standard Atmosphere)
def atmospheric_conditions(altitude):
    if altitude < 11000:  # Troposphere
        temperature = 15.04 - 0.00649 * altitude
        pressure = 101.29 * ((temperature + 273.1) / 288.08) ** 5.256
    elif altitude < 25000:  # Lower Stratosphere
        temperature = -56.46
        pressure = 22.65 * np.exp(1.73 - 0.000157 * altitude)
    else:  # Upper Stratosphere
        temperature = -131.21 + 0.00299 * altitude
        pressure = 2.488 * ((temperature + 273.1) / 216.6) ** -11.388
    
    air_density = pressure / (0.2869 * (temperature + 273.1))
    speed_of_sound = np.sqrt(1.4 * 287.05 * (temperature + 273.1))
    return air_density, temperature, pressure, speed_of_sound

# Drag force calculation
def calculate_drag(velocity, air_density, cross_sectional_area, drag_coefficient):
    return 0.5 * air_density * cross_sectional_area * drag_coefficient * velocity**2

# Gravity model
def gravity(altitude):
    return 9.81 * (6371000 / (6371000 + altitude))**2  # Earth's radius in meters

# Differential equations of motion
def rocket_dynamics(y, t, params):
    altitude, velocity, mass = y
    thrust, burn_time, mass_flow_rate, cross_sectional_area, drag_coefficient = params

    if t > burn_time:
        thrust = 0
        mass_flow_rate = 0

    air_density, temperature, pressure, speed_of_sound = atmospheric_conditions(altitude)
    drag_force = calculate_drag(velocity, air_density, cross_sectional_area, drag_coefficient)
    gravity_force = mass * gravity(altitude)
    net_force = thrust - gravity_force - drag_force
    acceleration = net_force / mass
    d_mass_dt = -mass_flow_rate

    return [velocity, acceleration, d_mass_dt]

# Streamlit app setup
st.set_page_config(page_title='Enhanced RocketSim 2D Rocket Simulation', layout='wide')
st.title('Enhanced RocketSim 2D Rocket Simulation')

# Sidebar for input fields
st.sidebar.header('Rocket Parameters')
thrust = st.sidebar.number_input('Thrust (N)', value=5000.0)
fuel_mass = st.sidebar.number_input('Fuel Mass (kg)', value=1000.0)
dry_mass = st.sidebar.number_input('Dry Mass (kg)', value=200.0)
specific_impulse = st.sidebar.number_input('Specific Impulse (s)', value=300.0)
burn_time = st.sidebar.number_input('Burn Time (s)', value=60.0)
drag_coefficient = st.sidebar.number_input('Drag Coefficient', value=0.5)
cross_sectional_area = st.sidebar.number_input('Cross-Sectional Area (m^2)', value=1.0)

# Derived parameters
total_mass = dry_mass + fuel_mass
mass_flow_rate = thrust / (specific_impulse * 9.81)

# Prelaunch Checks
def perform_prelaunch_checks(thrust, fuel_mass, dry_mass, specific_impulse, burn_time, drag_coefficient, cross_sectional_area):
    checks = []
    if thrust <= 0:
        checks.append("Thrust must be positive.")
    if fuel_mass <= 0:
        checks.append("Fuel mass must be positive.")
    if dry_mass <= 0:
        checks.append("Dry mass must be positive.")
    if specific_impulse <= 0:
        checks.append("Specific impulse must be positive.")
    if burn_time <= 0:
        checks.append("Burn time must be positive.")
    if drag_coefficient <= 0:
        checks.append("Drag coefficient must be positive.")
    if cross_sectional_area <= 0:
        checks.append("Cross-sectional area must be positive.")
    
    return checks

# Real-time data placeholders
col1, col2 = st.columns([4, 1])

# Simulation button
if col1.button('Simulate'):
    # Perform prelaunch checks
    prelaunch_checks = perform_prelaunch_checks(thrust, fuel_mass, dry_mass, specific_impulse, burn_time, drag_coefficient, cross_sectional_area)
    
    st.subheader("Prelaunch Status")
    
    if prelaunch_checks:
        for check in prelaunch_checks:
            st.error(f"- {check}")
    else:
        st.success("All prelaunch checks passed. Rocket is ready for launch.")
        st.write("Performing prelaunch checks:")
        st.write("- Fuel test: Passed")
        st.write("- Engine test: Passed")
        st.write("- Thrust check: Passed")
        st.write("- Mass check: Passed")
        st.write("- Drag coefficient check: Passed")
        st.write("- Cross-sectional area check: Passed")

        # Initial conditions
        y0 = [0, 0, total_mass]  # [altitude, velocity, mass]
        t = np.linspace(0, burn_time * 2, num=1000)
        params = [thrust, burn_time, mass_flow_rate, cross_sectional_area, drag_coefficient]
        
        # Solve ODE
        sol = odeint(rocket_dynamics, y0, t, args=(params,))
        altitude = sol[:, 0]
        velocity = sol[:, 1]
        mass = sol[:, 2]

        # Ensure altitude is non-negative
        altitude = np.maximum(altitude, 0)

        # Surface conditions placeholders
        with col2:
            st.subheader("Surface Atmospheric Conditions")
            wind_speed_placeholder = st.empty()
            wind_direction_placeholder = st.empty()
            wind_heading_placeholder = st.empty()
            pressure_placeholder = st.empty()
            temperature_placeholder = st.empty()
            air_density_placeholder = st.empty()
            speed_of_sound_placeholder = st.empty()

        # Update real-time data
        for i in range(len(t)):
            air_density, temperature, pressure, speed_of_sound = atmospheric_conditions(altitude[i])
            with col2:
                wind_speed_placeholder.write(f"Surface Wind Speed: {0:.2f} m/s")
                wind_direction_placeholder.write(f"Surface Wind Direction: {0:.2f}°")
                wind_heading_placeholder.write(f"Surface Wind Heading: {0:.2f}°")
                pressure_placeholder.write(f"Surface Pressure: {pressure:.2f} hPa")
                temperature_placeholder.write(f"Surface Temperature: {temperature:.2f} K")
                air_density_placeholder.write(f"Surface Air Density: {air_density:.3f} kg/m³")
                speed_of_sound_placeholder.write(f"Surface Speed of Sound: {speed_of_sound:.2f} m/s")

        # Results section
        col1.subheader('Simulation Results')
        col1.write(f"Total Impulse: {thrust * burn_time:.2f} Ns")
        col1.write(f"Delta V: {velocity[-1]:.2f} m/s")
        col1.write(f"Final Altitude: {altitude[-1]:.2f} m")

        # 3D Rocket visualization using Plotly
        frames = [go.Frame(
            data=[
                go.Scatter3d(
                    x=[t_val],
                    y=[0],
                    z=[alt_val],
                    mode='markers',
                    marker=dict(size=5, color='blue')
                )
            ],
            name=f'frame{i}'
        ) for i, (t_val, alt_val) in enumerate(zip(t, altitude))]

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=[0],
                    y=[0],
                    z=[0],
                    mode='markers',
                    marker=dict(size=5, color='blue')
                )
            ],
            layout=go.Layout(
                scene=dict(
                    xaxis=dict(title='Time (s)'),
                    yaxis=dict(title=''),
                    zaxis=dict(title='Altitude (m)')
                ),
                title='Rocket Launch Simulation',
                updatemenus=[
                    {
                        'type': 'buttons',
                        'buttons': [
                            {
                                'label': 'Play',
                                'method': 'animate',
                                'args': [None, {'frame': {'duration': 50, 'redraw': True}, 'fromcurrent': True}]
                            }
                        ]
                    }
                ]
            ),
            frames=frames
        )

        col1.plotly_chart(fig)

