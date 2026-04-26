import cv2
import torch
import pandas as pd
from datetime import datetime
import os
import time

# 🔥 LOAD YOLOv5 MODEL (IMPORTANT)
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='models\yolo_v5n.pt',
                       force_reload=False)

model.eval()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
)

marked = set()
cap = cv2.VideoCapture(0)

prev_time = 0
frame_count = 0
last_predictions = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 🔥 resize for speed
    frame = cv2.resize(frame, (240, 180))

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0,255,0), 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, 1.1, 3, minSize=(30,30)
    )

    frame_count += 1
    current_predictions = []

    run_model = (frame_count % 5 == 0)

    for idx, (x,y,w,h) in enumerate(faces):
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (128,128))

        if run_model:
            face = cv2.resize(face, (224, 224))

            # BGR → RGB
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            # HWC → CHW
            face = face.transpose(2, 0, 1)

            # normalize (0–1)
            face = face / 255.0

            # to tensor
            face = torch.tensor(face).float()

            # add batch dimension → (1,3,224,224)
            face = face.unsqueeze(0)

            # inference
            results = model(face)

            # 🔥 YOLOv5 classification output
            probs = torch.nn.functional.softmax(results[0], dim=0)
            conf, cls = torch.max(probs, 0)

            name = model.names[int(cls)]
            conf = float(conf)
            conf, cls = torch.max(probs, 0)

            name = model.names[int(cls)]
            conf = float(conf)

            if conf < 0.4:
                name = "Unknown"

            current_predictions.append((x,y,w,h,name,conf))
        else:
            if idx < len(last_predictions):
                current_predictions.append(last_predictions[idx])

    last_predictions = current_predictions

    for (x,y,w,h,name,conf) in current_predictions:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
        cv2.putText(frame,f"{name} ({conf:.2f})",
                    (x,y-5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,(0,255,0),1)

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

    cv2.imshow("YOLOv5 Attendance", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()