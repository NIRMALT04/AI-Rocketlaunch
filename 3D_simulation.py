import numpy as np
import plotly.graph_objects as go
import streamlit as st

# 3D Rocket Visualization Function
def create_3d_rocket(y_offset=0):
    fig = go.Figure()

    # Rocket Nose (Sphere)
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = 0.3 * np.outer(np.cos(u), np.sin(v))
    y = 0.3 * np.outer(np.sin(u), np.sin(v))
    z = 0.3 * np.outer(np.ones(np.size(u)), np.cos(v)) + 12 + y_offset
    fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale='gray', showscale=False, opacity=1))

    # Rocket Body (Cylinder)
    z_body = np.linspace(0, 12, 100) + y_offset
    theta = np.linspace(0, 2 * np.pi, 100)
    theta, z_body = np.meshgrid(theta, z_body)
    x_body = 0.3 * np.cos(theta)
    y_body = 0.3 * np.sin(theta)
    fig.add_trace(go.Surface(x=x_body, y=y_body, z=z_body, surfacecolor=np.full_like(z_body, 0.5), colorscale='gray', showscale=False, opacity=1))

    # Rocket Fins (Triangles)
    fin_height = 1.5
    fin_width = 1
    for angle in np.linspace(0, 2 * np.pi, 4, endpoint=False):
        x_fins = [0, fin_width * np.cos(angle), 0.1 * np.cos(angle)]
        y_fins = [0, fin_width * np.sin(angle), 0.1 * np.sin(angle)]
        z_fins = [0, 0, fin_height]
        fig.add_trace(go.Mesh3d(x=x_fins, y=y_fins, z=[z + y_offset for z in z_fins], color='gray', opacity=0.7))

    # Windows (Circular)
    window_radius = 0.05
    for z in [4, 6, 8]:
        window_x = window_radius * np.cos(theta[0])
        window_y = window_radius * np.sin(theta[0])
        fig.add_trace(go.Scatter3d(x=window_x, y=window_y, z=[z + y_offset] * len(theta[0]), mode='markers', marker=dict(size=5, color='white'), showlegend=False))

    # Thrusters (At the Base)
    thruster_radius = 0.1
    thruster_height = 0.5
    for angle in np.linspace(0, 2 * np.pi, 4, endpoint=False):
        x_thruster = [thruster_radius * np.cos(angle), 0]
        y_thruster = [thruster_radius * np.sin(angle), 0]
        z_thruster = [0 + y_offset, -thruster_height + y_offset]
        fig.add_trace(go.Scatter3d(x=x_thruster, y=y_thruster, z=z_thruster, mode='lines', line=dict(color='orange', width=4), showlegend=False))

    # Exhaust Flames (Effect at the Base)
    flame_z = np.linspace(0, -2, 30) + y_offset
    flame_r = np.linspace(0, 0.5, 30)
    flame_theta = np.linspace(0, 2 * np.pi, 30)
    flame_theta, flame_z = np.meshgrid(flame_theta, flame_z)
    flame_x = flame_r[:, np.newaxis] * np.cos(flame_theta)
    flame_y = flame_r[:, np.newaxis] * np.sin(flame_theta)
    fig.add_trace(go.Surface(x=flame_x, y=flame_y, z=flame_z, colorscale='Reds', opacity=0.5, showscale=False))

    # Labels
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[12.3 + y_offset],
        mode='text',
        text=['Nose Cone'],
        textposition='top center',
        showlegend=False
    ))
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[2 + y_offset],
        mode='text',
        text=['Window'],
        textposition='top center',
        showlegend=False
    ))

    fig.update_layout(scene=dict(
        xaxis=dict(nticks=4, range=[-2, 2]),
        yaxis=dict(nticks=4, range=[-2, 2]),
        zaxis=dict(nticks=4, range=[-2, 14 + y_offset]),
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5)
        )
    ))

    return fig

# Streamlit UI
st.title('Advanced Rocket Launch System')

if 'launch' not in st.session_state:
    st.session_state.launch = False

if 'y_offset' not in st.session_state:
    st.session_state.y_offset = 0

def launch_rocket():
    st.session_state.launch = True
    st.session_state.y_offset = 0

st.button('Launch Rocket', on_click=launch_rocket)

# Streamlit UI for 3D Visualization
st.subheader('3D Rocket Visualization')

# Simulate rocket launch by moving it upwards along the y-axis
if st.session_state.launch:
    frames = []
    for y in range(0, 101, 5):
        frame = create_3d_rocket(y_offset=y)
        frames.append(go.Frame(data=frame.data))
    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(updatemenus=[{
        'type': 'buttons',
        'showactive': False,
        'buttons': [{
            'label': 'Launch',
            'method': 'animate',
            'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}]
        }]
    }])
    st.plotly_chart(fig)
else:
    fig = create_3d_rocket()
    st.plotly_chart(fig)