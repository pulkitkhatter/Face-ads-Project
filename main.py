import cv2
import numpy as np
import os
import csv
import time
from datetime import datetime
import pygame  # Added for MP3 support

# ================= CONFIG =================
SAVE_FOLDER = "collected_faces"
LOG_FILE = "demographic_logs.csv"
COOLDOWN_SECONDS = 6
# Updated to your specific local path and file format
SOUND_FILE = "/Users/pulkitkhatter/Desktop/python project/alert.mp3" 

# Define the Alert Area (x_min, y_min, x_max, y_max) as percentages of the frame
ALERT_ZONE_PCT = (0.6, 0.2, 0.95, 0.8) # Right side of the screen

# Initialize pygame mixer for mp3 playback
pygame.mixer.init()

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
            "Image",
            "Alert_Triggered"
        ])

# ================= LOAD MODELS =================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

try:
    age_net = cv2.dnn.readNet("age_net.caffemodel", "deploy_age.prototxt")
    gender_net = cv2.dnn.readNet("gender_net.caffemodel", "deploy_gender.prototxt")
except Exception as e:
    print(f"Error loading DNN models: {e}")
    print("Ensure 'age_net.caffemodel', 'deploy_age.prototxt', etc. are in the script folder.")
    exit()

age_list = ['(0-2)','(4-6)','(8-12)','(15-20)', '(25-32)','(38-43)','(48-53)','(60-100)']
gender_list = ['Male','Female']

last_logged_time = 0
last_sound_time = 0
frame_counter = 0
start_time = time.time()

cap = cv2.VideoCapture(0)

def play_alert_sound():
    """Plays the MP3 alert music from the specified local path."""
    if os.path.exists(SOUND_FILE):
        try:
            # Check if music is already playing to avoid overlapping
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.load(SOUND_FILE)
                pygame.mixer.music.play()
        except Exception as e:
            print(f"Error playing sound: {e}")
    else:
        print(f"Sound file not found at: {SOUND_FILE}")

print("System Started... Press Q to Exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    fh, fw, _ = frame.shape
    frame_counter += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Calculate absolute coordinates for Alert Zone
    az_x1, az_y1 = int(ALERT_ZONE_PCT[0] * fw), int(ALERT_ZONE_PCT[1] * fh)
    az_x2, az_y2 = int(ALERT_ZONE_PCT[2] * fw), int(ALERT_ZONE_PCT[3] * fh)

    # Draw the Alert Area (Semi-transparent red overlay)
    overlay = frame.copy()
    cv2.rectangle(overlay, (az_x1, az_y1), (az_x2, az_y2), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
    cv2.rectangle(frame, (az_x1, az_y1), (az_x2, az_y2), (0, 0, 255), 2)
    cv2.putText(frame, "ALERT ZONE", (az_x1 + 5, az_y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    ad_text = "👋 Welcome! Please look at the screen."
    alert_triggered = False

    for (x, y, w, h) in faces:
        face_center_x = x + (w // 2)
        face_center_y = y + (h // 2)

        # Check if face center is inside Alert Zone
        in_alert_zone = (az_x1 < face_center_x < az_x2) and (az_y1 < face_center_y < az_y2)

        if in_alert_zone:
            alert_triggered = True
            # Play sound with a small cooldown so it doesn't stutter (2 seconds)
            if time.time() - last_sound_time > 2:
                play_alert_sound()
                last_sound_time = time.time()

        face_img = frame[y:y+h, x:x+w]
        face_gray = gray[y:y+h, x:x+w]

        # ================= AGE + GENDER =================
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227,227), (78.42, 87.76, 114.89), swapRB=False)

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

        # ================= SMILE + MOOD =================
        smiles = smile_cascade.detectMultiScale(face_gray, scaleFactor=1.7, minNeighbors=20)
        eyes = eye_cascade.detectMultiScale(face_gray)
        smile_count = len(smiles)
        eye_count = len(eyes)

        if smile_count > 0: mood = "Happy"
        elif eye_count == 0: mood = "Angry"
        elif eye_count == 1: mood = "Confused"
        elif smile_count == 0 and eye_count >= 2: mood = "Neutral"
        else: mood = "Serious"

        # ================= AD ENGINE =================
        if in_alert_zone:
            ad_text = "⚠️ RESTRICTED AREA! Please Step Back."
        elif gender == "Male" and age in ['(15-20)','(25-32)']:
            ad_text = "🔥 Gaming Laptop Mega Sale!"
        elif gender == "Female" and age in ['(15-20)','(25-32)']:
            ad_text = "💄 Beauty Products 40% OFF!"
        elif mood == "Happy":
            ad_text = "🎉 You're Smiling! Special Bonus Deal!"
        else:
            ad_text = "🛍️ Exclusive Offer For You!"

        # ================= LOGGING =================
        current_time = time.time()
        if current_time - last_logged_time > COOLDOWN_SECONDS:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = f"{SAVE_FOLDER}/{timestamp}.jpg"
            cv2.imwrite(image_path, face_img)

            with open(LOG_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, gender, gender_conf, age, age_conf, mood, smile_count, image_path, in_alert_zone])
            last_logged_time = current_time

        # Draw box and labels
        color = (0, 0, 255) if in_alert_zone else (0, 255, 0)
        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
        cv2.putText(frame,f"{gender} {age}", (x,y-40), cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
        cv2.putText(frame,f"Mood: {mood}", (x,y-20), cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

    # ================= INFO PANEL =================
    fps = round(frame_counter / (time.time() - start_time), 2)
    cv2.rectangle(frame,(0,0),(frame.shape[1],90),(40,40,40),-1)
    cv2.putText(frame,f"FPS: {fps} | Faces: {len(faces)}", (20,30), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
    cv2.putText(frame,ad_text, (20,70), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

    cv2.imshow("Advanced Smart AI Ad Engine + Security", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()