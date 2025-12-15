import cv2
import os

path = "C:/Users/Charles/Downloads/853889-hd_1920_1080_25fps.mp4"

print(f"Checking file: {path}")
if not os.path.exists(path):
    print("ERROR: File does not exist!")
else:
    print(f"File exists. Size: {os.path.getsize(path)} bytes")

cap = cv2.VideoCapture(path)

if not cap.isOpened():
    print("ERROR: cv2.VideoCapture could not open the file.")
else:
    print(f"SUCCESS: cv2.VideoCapture opened the file. Backend: {cap.getBackendName()}")
    
    ret, frame = cap.read()
    if ret:
        print(f"SUCCESS: Read first frame. Shape: {frame.shape}")
    else:
        print("ERROR: Failed to read first frame (ret=False).")

cap.release()
