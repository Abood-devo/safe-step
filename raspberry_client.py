import cv2
import socket
import pickle
import struct
import RPi.GPIO as GPIO
import time
import pygame
from threading import Thread

# Network settings
SERVER_HOST = 'your_server_ip'  # Replace with your processing machine's IP
SERVER_PORT = 8000

# Servo settings
SERVO_PIN = 18
SERVO_POSITIONS = {
    'left': 0,
    'front': 90,
    'right': 180
}

# Audio settings
pygame.mixer.init()
AUDIO_FILES = {
    'person_close': 'audio/person_close.wav',
    'person_far': 'audio/person_far.wav',
    'car_close': 'audio/car_close.wav',
    'car_far': 'audio/car_far.wav'
}

class CameraSystem:
    def __init__(self):
        # Initialize camera
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Initialize servo
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(SERVO_PIN, GPIO.OUT)
        self.servo = GPIO.PWM(SERVO_PIN, 50)  # 50Hz frequency
        self.servo.start(0)
        self.current_direction = 'front'
        self.set_servo_position('front')
        
        # Initialize network
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((SERVER_HOST, SERVER_PORT))
        
    def set_servo_position(self, direction):
        if direction in SERVO_POSITIONS and direction != self.current_direction:
            duty = 2 + (SERVO_POSITIONS[direction] / 18)
            self.servo.ChangeDutyCycle(duty)
            time.sleep(0.3)
            self.current_direction = direction
            
    def play_audio(self, object_class, proximity):
        if proximity > 0.3:  # Close proximity threshold
            audio_key = f"{object_class}_close"
        else:
            audio_key = f"{object_class}_far"
            
        if audio_key in AUDIO_FILES:
            pygame.mixer.music.load(AUDIO_FILES[audio_key])
            pygame.mixer.music.play()
            
    def process_detection(self, detection):
        if detection:
            self.set_servo_position(detection['direction'])
            self.play_audio(detection['class'], detection['proximity'])
            
    def run(self):
        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    break
                    
                # Serialize and send frame
                data = pickle.dumps(frame)
                self.client_socket.sendall(struct.pack("L", len(data)) + data)
                
                # Receive detection results
                data = b""
                payload_size = struct.calcsize("L")
                
                while len(data) < payload_size:
                    data += self.client_socket.recv(4096)
                    
                packed_msg_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack("L", packed_msg_size)[0]
                
                while len(data) < msg_size:
                    data += self.client_socket.recv(4096)
                    
                detection = pickle.loads(data[:msg_size])
                self.process_detection(detection)
                
        finally:
            self.cleanup()
            
    def cleanup(self):
        self.camera.release()
        self.servo.stop()
        GPIO.cleanup()
        self.client_socket.close()

if __name__ == "__main__":
    system = CameraSystem()
    system.run()