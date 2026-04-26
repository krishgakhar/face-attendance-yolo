import cv2
import os
import time

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
)

def capture(name, mode="train", count=90):
    path = f"dataset/{mode}/{name}"
    os.makedirs(path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    i = 0
    last_time = 0

    while i < count:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(40,40)
        )

        for (x,y,w,h) in faces:
            if time.time() - last_time > 0.3:
                face = frame[y:y+h, x:x+w]
                face = cv2.resize(face, (224,224))

                cv2.imwrite(f"{path}/{i}.jpg", face)
                i += 1
                last_time = time.time()

            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        cv2.putText(frame, f"{mode}:{i}/{count}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        cv2.imshow("Capture", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    name = input("Enter name: ")

    capture(name, "train", 90)
    capture(name, "val", 10)