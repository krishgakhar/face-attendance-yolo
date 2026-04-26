import cv2
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import os
import time

prev_time=0

model = YOLO(r"C:\Users\KRISH\Desktop\yolov5\runs\classify\train\weights\best.pt")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
)

marked = set()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640,480))
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(40,40)
    )

    print("Faces detected:", len(faces))

    for (x,y,w,h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (224,224))

        results = model(face)

        probs = results[0].probs
        name = results[0].names[probs.top1]
        conf = float(probs.top1conf)

        if conf < 0.65:
            name = "Unknown"

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,f"{name} ({conf:.2f})",
                    (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,(0,255,0),2)

        # attendance
        if name != "Unknown" and name not in marked:
            marked.add(name)

            os.makedirs("attendance", exist_ok=True)

            df = pd.DataFrame([{
                "Name": name,
                "Time": datetime.now()
            }])

            file = "attendance/attendance.csv"

            if not os.path.exists(file):
                df.to_csv(file, index=False)
            else:
                df.to_csv(file, mode='a', header=False, index=False)

    cv2.imshow("Attendance", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()