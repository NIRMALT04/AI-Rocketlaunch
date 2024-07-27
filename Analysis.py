import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import time
import socket


np.random.seed(42)
num_samples = 1000
data = {
    'temperature': np.random.normal(25, 5, num_samples),
    'pressure': np.random.normal(1013, 10, num_samples),
    'vibration': np.random.normal(0.5, 0.1, num_samples),
    'fuel_level': np.random.uniform(0, 100, num_samples),
    'battery_voltage': np.random.normal(12, 0.5, num_samples),
}
failures = (
    (data['temperature'] > 35) |
    (data['pressure'] < 1000) |
    (data['vibration'] > 0.7) |
    (data['fuel_level'] < 10) |
    (data['battery_voltage'] < 11)
)
data['failure'] = failures.astype(int)
df = pd.DataFrame(data)
df.to_csv('synthetic_sensor_data.csv', index=False)
print("Synthetic dataset created and saved to 'synthetic_sensor_data.csv'")


data = pd.read_csv('synthetic_sensor_data.csv')
X = data.drop('failure', axis=1)
y = data['failure']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")


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

pid = PIDController(1.0, 0.1, 0.05)
setpoint = 100
measurement = 80
control_signal = pid.update(setpoint, measurement)
print(f"Control signal: {control_signal}")


from rocketpy import Environment

Env = Environment(latitude=32.990254, longitude=-106.974998, elevation=1400)
Env.set_date((2024, 7, 27, 6))  
Env.set_atmospheric_model(type='Forecast', file='GFS')
Env.info()

wind_speed = Env.wind.speed
wind_direction = Env.wind.direction
print(f"Wind speed: {wind_speed} m/s, Wind direction: {wind_direction} degrees")


class LaunchSequencer:
    def __init__(self):
        self.countdown = 10

    def start_countdown(self):
        while self.countdown > 0:
            print(f"T-minus {self.countdown} seconds")
            time.sleep(1)
            self.countdown -= 1
        print("Launch!")

sequencer = LaunchSequencer()


def send_telemetry(data):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('ground_control_ip', 12345))
    s.sendall(data.encode())
    s.close()

telemetry_data = "altitude=1000;speed=5400"
send_telemetry(telemetry_data)


class RocketController:
    def __init__(self):
        self.pre_launch_checks_done = False
        self.fueling_done = False
        self.positioning_done = False

    def run_pre_launch_checks(self):
        
        self.pre_launch_checks_done = True

    def start_fueling(self):
        
        self.fueling_done = True

    def position_rocket(self):
        
        self.positioning_done = True

    def launch(self):
        if self.pre_launch_checks_done and self.fueling_done and self.positioning_done:
            sequencer.start_countdown()
        else:
            print("Cannot launch, not all systems are ready")

controller = RocketController()
controller.run_pre_launch_checks()
controller.start_fueling()
controller.position_rocket()
controller.launch()
