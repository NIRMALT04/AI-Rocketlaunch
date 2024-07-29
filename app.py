from flask import Flask, jsonify, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import socket
import os
from rocketpy import Environment
import subprocess

app = Flask(__name__)

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

@app.route('/pid', methods=['POST'])
def pid_controller():
    data = request.get_json()
    kp = data['kp']
    ki = data['ki']
    kd = data['kd']
    setpoint = data['setpoint']
    measurement = data['measurement']
    pid = PIDController(kp, ki, kd)
    control_signal = pid.update(setpoint, measurement)
    return jsonify(control_signal=control_signal)

@app.route('/train_model', methods=['GET'])
def train_model():
    data = pd.read_csv('synthetic_sensor_data.csv')
    X = data.drop('failure', axis=1)
    y = data['failure']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return jsonify(accuracy=accuracy)

@app.route('/send_telemetry', methods=['POST'])
def send_telemetry():
    data = request.get_json()
    telemetry_data = data['telemetry']
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('ground_control_ip', 12345))  # Ensure 'ground_control_ip' is replaced with the actual IP
        s.sendall(telemetry_data.encode())
        s.close()
        return jsonify(status="success", message="Telemetry data sent!")
    except Exception as e:
        return jsonify(status="error", message=f"Failed to send telemetry data: {e}")

@app.route('/rocket_environment', methods=['GET'])
def rocket_environment():
    try:
        env = Environment(latitude=32.990254, longitude=-106.974998, elevation=1400)
        valid_date = (2024, 7, 29, 6)  # Example: Adjust to a valid date within the range
        env.set_date(valid_date)
        env.set_atmospheric_model(type='Forecast', file='GFS')
        wind_speed = np.random.uniform(0, 20)  # Random wind speed between 0 and 20 m/s
        wind_direction = np.random.uniform(0, 360)  # Random wind direction between 0 and 360 degrees
        return jsonify(wind_speed=wind_speed, wind_direction=wind_direction)
    except ValueError as e:
        return jsonify(error=str(e))
    except Exception as e:
        return jsonify(error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
