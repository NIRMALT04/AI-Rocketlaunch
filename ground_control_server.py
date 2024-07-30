import socket
import threading

class GroundControlServer:
    def __init__(self, host='127.0.0.1', port=12345):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.host, self.port))
        self.sock.listen(5)
        print("Ground Control Server is listening on port", self.port)
        self.clients = []

    def start(self):
        while True:
            client_socket, addr = self.sock.accept()
            print(f"Connection established with {addr}")
            self.clients.append(client_socket)
            threading.Thread(target=self.handle_client, args=(client_socket,), daemon=True).start()

    def handle_client(self, client_socket):
        while True:
            try:
                data = client_socket.recv(1024).decode()
                if data:
                    print(f"Telemetry data received: {data}")
                    self.send_command(client_socket, "Adjust course by 5 degrees")
            except Exception as e:
                print(f"Client disconnected: {e}")
                self.clients.remove(client_socket)
                client_socket.close()
                break

    def send_command(self, client_socket, command):
        try:
            client_socket.sendall(command.encode())
            print(f"Command sent: {command}")
        except Exception as e:
            print(f"Failed to send command: {e}")
