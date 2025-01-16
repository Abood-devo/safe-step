from ultralytics import YOLO
import cv2
import numpy as np
import socket
import pickle
import struct
import threading
import time

# YOLO model initialization
model = YOLO('yolo11n.pt')

# Network settings
HOST = '0.0.0.0'  # Listen on all available interfaces
PORT = 8000

def process_frame(frame):
    # Run YOLO detection
    results = model(frame)
    
    # Process results
    for result in results:
        boxes = result.boxes
        if len(boxes) > 0:
            # Get closest detection (largest box)
            areas = [box.xywh[0][2] * box.xywh[0][3] for box in boxes]
            largest_idx = np.argmax(areas)
            box = boxes[largest_idx]
            
            # Calculate position relative to frame center
            frame_center = frame.shape[1] / 2
            box_center = box.xywh[0][0]
            
            # Determine direction
            if box_center < frame_center - frame.shape[1]/4:
                direction = "left"
            elif box_center > frame_center + frame.shape[1]/4:
                direction = "right"
            else:
                direction = "front"
                
            # Calculate proximity (based on box size)
            proximity = areas[largest_idx] / (frame.shape[0] * frame.shape[1])
            
            return {
                "direction": direction,
                "proximity": proximity,
                "class": result.names[int(box.cls[0])]
            }
    return None

def handle_client(client_socket):
    data = b""
    payload_size = struct.calcsize("L")
    
    while True:
        while len(data) < payload_size:
            data += client_socket.recv(4096)
        
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("L", packed_msg_size)[0]
        
        while len(data) < msg_size:
            data += client_socket.recv(4096)
            
        frame_data = data[:msg_size]
        data = data[msg_size:]
        
        frame = pickle.loads(frame_data)
        result = process_frame(frame)
        
        # Send back detection results
        if result:
            response = pickle.dumps(result)
            client_socket.send(struct.pack("L", len(response)) + response)

def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(10)
    print(f"Server listening on {HOST}:{PORT}")
    
    while True:
        client_socket, addr = server_socket.accept()
        print(f"Connected to {addr}")
        client_thread = threading.Thread(target=handle_client, args=(client_socket,))
        client_thread.start()

if __name__ == "__main__":
    start_server()