import socket
import threading
import streamlit as st

class RocketClient:
    def __init__(self, host='127.0.0.1', port=12345):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.telemetry_data = "altitude=1000;speed=5400"

    def connect(self):
        try:
            self.sock.connect((self.host, self.port))
            st.success("Connected to ground control!")
            threading.Thread(target=self.receive_commands, daemon=True).start()
        except ConnectionRefusedError:
            st.error("Connection refused by ground control. Ensure the server is running and listening on the port.")
        except Exception as e:
            st.error(f"Failed to connect to ground control: {e}")

    def send_telemetry(self):
        try:
            self.sock.sendall(self.telemetry_data.encode())
            st.success("Telemetry data sent!")
        except Exception as e:
            st.error(f"Failed to send telemetry data: {e}")

    def receive_commands(self):
        while True:
            try:
                command = self.sock.recv(1024).decode()
                if command:
                    st.write(f"Command received from ground control: {command}")
            except Exception as e:
                st.error(f"Failed to receive command: {e}")
                break
