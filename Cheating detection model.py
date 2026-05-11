import cv2
import numpy as np
import time
from sklearn.datasets import fetch_lfw_people


print("Fetching LFW Dataset (This might take a moment on the first run)...")
lfw_dataset = fetch_lfw_people(min_faces_per_person=70, resize=0.5)
lfw_faces = [np.uint8(img * 255) for img in lfw_dataset.images]
lfw_labels = list(np.array(lfw_dataset.target, dtype=np.int32))
target_names = list(lfw_dataset.target_names)

user_label_id = len(target_names)
target_names.append("Authorized Student")
print(f"LFW Loaded! {len(lfw_faces)} background faces ready.")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()


is_registered = False
registration_frames = []
REQUIRED_FRAMES = 90  
last_eye_seen_time = time.time() 

def get_all_faces(gray_frame, frame_w):
    faces = list(face_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50)))
    face_type = "frontal"
    if len(faces) == 0:
        profiles = list(profile_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50)))
        if len(profiles) > 0:
            faces = profiles
            face_type = "profile"
        else:
            flipped_gray = cv2.flip(gray_frame, 1)
            left_profiles = profile_cascade.detectMultiScale(flipped_gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))
            for (x, y, w, h) in left_profiles:
                faces.append((frame_w - x - w, y, w, h))
            if len(faces) > 0:
                face_type = "profile"
    return faces, face_type


cap = cv2.VideoCapture(0)
print("Starting Camera... Get ready for 3-stage registration.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame_h, frame_w, _ = frame.shape
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    warnings = []
    
    detected_faces, face_type = get_all_faces(gray_frame, frame_w)

   
    if not is_registered:
        progress = len(registration_frames)
        if progress < 30:
            instruction = "1/3: Look STRAIGHT at camera..."
            color = (0, 255, 255)
        elif progress < 60:
            instruction = "2/3: Turn head slightly LEFT..."
            color = (255, 100, 100)
        else:
            instruction = "3/3: Turn head slightly RIGHT..."
            color = (100, 255, 100)

        cv2.putText(frame, f"REGISTRATION: {instruction} ({progress}/{REQUIRED_FRAMES})", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if len(detected_faces) == 1:
            x, y, w, h = detected_faces[0]
            roi_gray = gray_frame[y:y+h, x:x+w]
            registration_frames.append(roi_gray)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            if len(registration_frames) >= REQUIRED_FRAMES:
                cv2.putText(frame, "TRAINING AI ON LFW + YOUR FACE... PLEASE WAIT", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow('Pure OpenCV Proctoring', frame)
                cv2.waitKey(1) 
                
                user_labels = [user_label_id] * len(registration_frames)
                all_faces = lfw_faces + registration_frames
                all_labels = np.array(lfw_labels + user_labels, dtype=np.int32)
                
                face_recognizer.train(all_faces, all_labels)
                is_registered = True
                last_eye_seen_time = time.time() 
                print("Training Complete! Monitoring started.")
                
        elif len(detected_faces) > 1:
            cv2.putText(frame, "ERROR: Multiple faces!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

   
    else:
        person_count = len(detected_faces)
        eyes_detected_this_frame = False
        
        # 1. Identity Check
        for (x, y, w, h) in detected_faces:
            roi_gray = gray_frame[y:y+h, x:x+w]
            if roi_gray.size > 0:
                label_id, confidence = face_recognizer.predict(roi_gray)
                
                if label_id == user_label_id and confidence < 90:
                    display_text, box_color = target_names[label_id], (0, 255, 0)
                else:
                    display_text = "UNAUTHORIZED: " + target_names.get(label_id, "Unknown") if label_id != user_label_id else "UNAUTHORIZED"
                    box_color = (0, 0, 255)
                    warnings.append("UNAUTHORIZED PERSON")
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                cv2.putText(frame, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

                # 2. EYE & PUPIL TRACKING
                if face_type == "frontal":
                    eye_band_y_start = y + int(h * 0.20)
                    eye_band_y_end = y + int(h * 0.55)
                    roi_eye_band = gray_frame[eye_band_y_start:eye_band_y_end, x:x + w]
                    min_eye_size = (int(w * 0.12), int(h * 0.12))
                    detected_eyes = eye_cascade.detectMultiScale(roi_eye_band, scaleFactor=1.1, minNeighbors=7, minSize=min_eye_size)
                    
                    if len(detected_eyes) > 0:
                        eyes_detected_this_frame = True
                        for (ex, ey, ew, eh) in detected_eyes:
                            cv2.rectangle(frame, (x + ex, eye_band_y_start + ey), 
                                          (x + ex + ew, eye_band_y_start + ey + eh), (0, 255, 0), 1)
                            
                            # --- NEW: PUPIL TRACKING ---
                            eye_roi_gray = roi_eye_band[ey:ey+eh, ex:ex+ew]
                            
                            # Apply blur to remove skin texture noise
                            eye_blur = cv2.GaussianBlur(eye_roi_gray, (7, 7), 0)
                            
                            # Find the darkest pixels (Threshold might need adjusting based on your room lighting)
                            # 45 is a good default. If it can't find your pupil, raise it to 60.
                            _, thresh = cv2.threshold(eye_blur, 45, 255, cv2.THRESH_BINARY_INV)
                            
                            # Find blobs of dark pixels
                            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                            
                            if contours:
                                # Assume the largest dark spot is the pupil
                                pupil_cnt = max(contours, key=cv2.contourArea)
                                M = cv2.moments(pupil_cnt)
                                
                                if M['m00'] != 0:
                                    cx = int(M['m10']/M['m00']) # X coordinate of pupil
                                    cy = int(M['m01']/M['m00']) # Y coordinate of pupil
                                    
                                    # Draw a red dot directly on the pupil!
                                    cv2.circle(frame, (x + ex + cx, eye_band_y_start + ey + cy), 3, (0, 0, 255), -1)
                                    
                                    # Calculate where the pupil is horizontally (0.0 is far left, 1.0 is far right)
                                    x_ratio = cx / ew
                                    
                                    # If the pupil leaves the 35% - 65% center zone, trigger the alert
                                    if x_ratio < 0.35 or x_ratio > 0.65:
                                        warnings.append("EYES DIVERTED (NOT CENTERED)")

        if eyes_detected_this_frame:
            last_eye_seen_time = time.time() 
        elif (time.time() - last_eye_seen_time) > 2.0:
            warnings.append("NO EYES DETECTED")

        if person_count == 0: warnings.append("NO FACE DETECTED")
        elif person_count > 1: warnings.append("MULTIPLE PEOPLE DETECTED")

        # 3. Arm Detection (Desk Zone)
        desk_y_start = int(frame_h * 0.60)
        desk_zone_hsv = cv2.cvtColor(frame[desk_y_start:frame_h, 0:frame_w], cv2.COLOR_BGR2HSV)
        cv2.line(frame, (0, desk_y_start), (frame_w, desk_y_start), (255, 0, 0), 1)

        lower_skin, upper_skin = np.array([0, 20, 70], dtype=np.uint8), np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(desk_zone_hsv, lower_skin, upper_skin)
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        arm_detected = any(cv2.contourArea(c) > 6000 for c in contours)
        if not arm_detected: warnings.append("ARMS NOT VISIBLE IN DESK ZONE")

        # --- UI: Display Warnings ---
        y_offset = 30
        for warning in set(warnings): 
            cv2.putText(frame, f"WARNING: {warning}", (20, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 30

    cv2.imshow('Pure OpenCV Proctoring', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()