import cv2
import numpy as np
import os
import csv
import time
from datetime import datetime

# ================= CONFIG =================
SAVE_FOLDER = "collected_faces"
LOG_FILE = "demographic_logs.csv"
COOLDOWN_SECONDS = 6

if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Timestamp",
            "Gender","Gender_Conf",
            "Age","Age_Conf",
            "Mood",
            "Smile_Count",
            "Image"
        ])

# ================= LOAD MODELS =================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_smile.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

age_net = cv2.dnn.readNet("age_net.caffemodel", "deploy_age.prototxt")
gender_net = cv2.dnn.readNet("gender_net.caffemodel", "deploy_gender.prototxt")

age_list = ['(0-2)','(4-6)','(8-12)','(15-20)',
            '(25-32)','(38-43)','(48-53)','(60-100)']
gender_list = ['Male','Female']

last_logged_time = 0
frame_counter = 0
start_time = time.time()

cap = cv2.VideoCapture(0)

print("System Started... Press Q to Exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # ✅ Default ad (prevents crash if no face)
    ad_text = "👋 Welcome! Please look at the screen."

    for (x, y, w, h) in faces:

        face_img = frame[y:y+h, x:x+w]
        face_gray = gray[y:y+h, x:x+w]

        # ================= AGE + GENDER =================
        blob = cv2.dnn.blobFromImage(
            face_img, 1.0, (227,227),
            (78.4263377603,87.7689143744,114.895847746),
            swapRB=False
        )

        # AGE
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age_index = age_preds[0].argmax()
        age = age_list[age_index]
        age_conf = round(float(age_preds[0][age_index])*100, 2)

        # GENDER
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender_index = gender_preds[0].argmax()
        gender = gender_list[gender_index]
        gender_conf = round(float(gender_preds[0][gender_index])*100, 2)

        # ================= SMILE + EYE =================
        smiles = smile_cascade.detectMultiScale(
            face_gray, scaleFactor=1.7, minNeighbors=20
        )

        eyes = eye_cascade.detectMultiScale(face_gray)

        smile_count = len(smiles)
        eye_count = len(eyes)

        # ================= MOOD ESTIMATION =================
        if smile_count > 0:
            mood = "Happy"
        elif eye_count == 0:
            mood = "Angry"
        elif eye_count == 1:
            mood = "Confused"
        elif smile_count == 0 and eye_count >= 2:
            mood = "Neutral"
        else:
            mood = "Serious"

        # ================= AD ENGINE =================
        if gender == "Male" and age in ['(15-20)','(25-32)']:
            ad_text = "🔥 Gaming Laptop Mega Sale!"
        elif gender == "Female" and age in ['(15-20)','(25-32)']:
            ad_text = "💄 Beauty Products 40% OFF!"
        elif mood == "Happy":
            ad_text = "🎉 You're Smiling! Special Bonus Deal!"
        elif mood == "Angry":
            ad_text = "😌 Relaxation Products - 30% OFF!"
        else:
            ad_text = "🛍️ Exclusive Offer For You!"

        # ================= COOLDOWN LOGGING =================
        current_time = time.time()

        if current_time - last_logged_time > COOLDOWN_SECONDS:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = f"{SAVE_FOLDER}/{timestamp}.jpg"
            cv2.imwrite(image_path, face_img)

            with open(LOG_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp,
                    gender, gender_conf,
                    age, age_conf,
                    mood,
                    smile_count,
                    image_path
                ])

            last_logged_time = current_time

        # ================= DRAW FACE INFO =================
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        cv2.putText(frame,f"Gender: {gender} ({gender_conf}%)",
                    (x,y-60),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

        cv2.putText(frame,f"Age: {age} ({age_conf}%)",
                    (x,y-40),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

        cv2.putText(frame,f"Mood: {mood}",
                    (x,y-20),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    # ================= GLOBAL INFO PANEL =================
    fps = round(frame_counter / (time.time() - start_time), 2)

    cv2.rectangle(frame,(0,0),(frame.shape[1],90),(40,40,40),-1)

    cv2.putText(frame,f"FPS: {fps}",
                (20,30),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

    cv2.putText(frame,f"Faces Detected: {len(faces)}",
                (160,30),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

    cv2.putText(frame,ad_text,
                (20,70),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)

    cv2.imshow("Advanced Smart AI Ad Engine",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()