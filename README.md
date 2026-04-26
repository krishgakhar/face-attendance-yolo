# Face Attendance System

## Pipeline
Webcam → Haar Cascade → Face Detection → Face Crop → Classification → Attendance CSV

## Models Used
- YOLOv5n
- YOLOv8n-cls (Best)
- MobileNetV2

## Installation
pip install -r requirements.txt

## Note for Jetson Nano
Install PyTorch manually from NVIDIA (default pip install will NOT work)

## Run
python src/recognize_v8n.py

## Output
attendance/attendance.csv