import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import pandas as pd
from datetime import datetime
import os
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔥 CLASS NAMES (MATCH TRAINING OUTPUT EXACTLY)
class_names = ["Arham","Krish","Priyanka"]   # CHANGE THIS

# 🔥 MODEL
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))

model.load_state_dict(torch.load("models/mobilenetv2_best.pth", map_location=device))
model.to(device)
model.eval()

# 🔥 TRANSFORM (MUST MATCH TRAINING)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
)

marked = set()
cap = cv2.VideoCapture(0)

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (320,240))

    # FPS
    curr_time = time.time()
    fps = 1/(curr_time-prev_time) if prev_time!=0 else 0
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}",
                (10,20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,(0,255,0),1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.1,3,minSize=(40,40))

    for (x,y,w,h) in faces:

        if w < 40 or h < 40:
            continue

        face = frame[y:y+h, x:x+w]

        img = transform(face).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img)
            probs = torch.softmax(outputs, dim=1)
            conf, cls = torch.max(probs, 1)

        name = class_names[int(cls)]
        conf = float(conf)

        print(name, conf)  # DEBUG

        if conf < 0.4:
            name = "Unknown"

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

    cv2.imshow("MobileNet", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()