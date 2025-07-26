import cv2
import numpy as np
import os
from datetime import datetime
import time

# Load multiple face detection cascades for better detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

# Load the face recognition model with very lenient parameters
recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=1,  # Smaller radius for more lenient matching
    neighbors=8,  # Fewer neighbors for more lenient matching
    grid_x=4,  # Smaller grid for more lenient matching
    grid_y=4,  # Smaller grid for more lenient matching
    threshold=90  # Even more lenient threshold
)

# Create Attendance.csv if it doesn't exist
attendance_file = 'Attendance.csv'
if not os.path.exists(attendance_file):
    try:
        with open(attendance_file, 'w', newline='') as f:
            f.write('Name,Date,Time\n')  # Changed header to separate Date and Time
        print(f"Created new attendance file: {attendance_file}")
    except Exception as e:
        print(f"Error creating attendance file: {str(e)}")
        exit()

path = 'Training_images'
if not os.path.exists(path):
    os.makedirs(path)
    print(f"Created {path} directory. Please add training images there.")

images = []
classNames = []
myList = os.listdir(path)
print("Training images found:", myList)

# Prepare training data
faces = []
ids = []

def preprocess_face(face_img):
    # Resize to a smaller size for more lenient matching
    face_img = cv2.resize(face_img, (100, 100))
    
    # Convert to grayscale if not already
    if len(face_img.shape) == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    # Simple normalization
    face_img = cv2.normalize(face_img, None, 0, 255, cv2.NORM_MINMAX)
    
    return face_img

def augment_face(face_img):
    augmented_faces = []
    
    # Original face
    augmented_faces.append(face_img)
    
    # Horizontal flip
    augmented_faces.append(cv2.flip(face_img, 1))
    
    # Simple rotation
    center = (face_img.shape[1] // 2, face_img.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 10, 1.0)
    rotated = cv2.warpAffine(face_img, rotation_matrix, (face_img.shape[1], face_img.shape[0]))
    augmented_faces.append(rotated)
    
    return augmented_faces

print("Processing training images...")
for cl in myList:
    try:
        curImg = cv2.imread(f'{path}/{cl}')
        if curImg is None:
            print(f"Warning: Could not read image {cl}")
            continue
            
        gray = cv2.cvtColor(curImg, cv2.COLOR_BGR2GRAY)
        
        # Try both frontal and profile face detection with more lenient parameters
        face_rects = face_cascade.detectMultiScale(gray, 1.05, 3, minSize=(20, 20))
        profile_rects = profile_cascade.detectMultiScale(gray, 1.05, 3, minSize=(20, 20))
        
        if len(face_rects) == 0 and len(profile_rects) == 0:
            print(f"Warning: No face detected in {cl}")
            continue
            
        # Process all detected faces
        for rects in [face_rects, profile_rects]:
            for (x, y, w, h) in rects:
                # Add padding around the face
                padding = int(min(w, h) * 0.1)
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(gray.shape[1] - x, w + 2 * padding)
                h = min(gray.shape[0] - y, h + 2 * padding)
                
                face_roi = gray[y:y+h, x:x+w]
                face_roi = preprocess_face(face_roi)
                augmented_faces = augment_face(face_roi)
                
                for aug_face in augmented_faces:
                    faces.append(aug_face)
                    ids.append(len(classNames))
        
        classNames.append(os.path.splitext(cl)[0])
        print(f"Successfully processed {cl} with {len(augmented_faces)} augmented samples")
        
    except Exception as e:
        print(f"Error processing {cl}: {str(e)}")
        continue

if len(faces) == 0:
    print("No valid training data found. Please add proper face images to the Training_images folder.")
    exit()

print(f"Total training samples: {len(faces)}")

# Train the recognizer
try:
    recognizer.train(faces, np.array(ids))
    print('Training Complete')
except Exception as e:
    print(f"Error during training: {str(e)}")
    exit()

# Keep track of last attendance mark to prevent duplicate entries
last_attendance = {}
attendance_cooldown = 5  # seconds between attendance marks

def markAttendance(name):
    try:
        current_time = time.time()
        
        # Check if we've marked attendance for this person recently
        if name in last_attendance:
            time_since_last = current_time - last_attendance[name]
            if time_since_last < attendance_cooldown:
                print(f"Waiting {attendance_cooldown - int(time_since_last)} seconds before next attendance mark")
                return
        
        # Read existing attendance records
        existing_records = []
        if os.path.exists(attendance_file):
            with open(attendance_file, 'r') as f:
                existing_records = f.readlines()
        
        # Check if attendance already marked for today
        today = datetime.now().strftime('%Y-%m-%d')
        name_marked = False
        for record in existing_records:
            if name in record and today in record:
                name_marked = True
                break
        
        if not name_marked:
            # Mark new attendance
            now = datetime.now()
            date_str = now.strftime('%Y-%m-%d')
            time_str = now.strftime('%H:%M:%S')
            
            with open(attendance_file, 'a', newline='') as f:
                f.write(f'{name},{date_str},{time_str}\n')  # Changed format to separate columns
            
            print(f"Marked attendance for {name} on {date_str} at {time_str}")
            last_attendance[name] = current_time
        else:
            print(f"Attendance already marked for {name} today")
            
    except Exception as e:
        print(f"Error marking attendance: {str(e)}")
        # Try to create the file if it doesn't exist
        try:
            with open(attendance_file, 'w', newline='') as f:
                f.write('Name,Date,Time\n')  # Changed header to separate Date and Time
            print(f"Created new attendance file: {attendance_file}")
        except Exception as e2:
            print(f"Error creating attendance file: {str(e2)}")

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Set balanced resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

# Variables for FPS calculation and frame skipping
prev_time = 0
curr_time = 0
frame_count = 0
skip_frames = 1  # Process every 2nd frame

print("Starting face recognition...")
print("Press 'q' to quit")

while True:
    success, img = cap.read()
    if not success:
        print("Error: Could not read frame from webcam")
        break
    
    frame_count += 1
    if frame_count % (skip_frames + 1) != 0:
        continue
        
    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
    prev_time = curr_time
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Try both frontal and profile face detection with more lenient parameters
    faces = face_cascade.detectMultiScale(gray, 1.05, 3, minSize=(20, 20))
    profiles = profile_cascade.detectMultiScale(gray, 1.05, 3, minSize=(20, 20))
    
    # Process all detected faces
    for rects in [faces, profiles]:
        for (x, y, w, h) in rects:
            try:
                # Add padding around the face
                padding = int(min(w, h) * 0.1)
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(gray.shape[1] - x, w + 2 * padding)
                h = min(gray.shape[0] - y, h + 2 * padding)
                
                face_roi = gray[y:y+h, x:x+w]
                face_roi = preprocess_face(face_roi)
                
                # Predict face
                id_, confidence = recognizer.predict(face_roi)
                
                # Draw rectangle around face
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                if confidence < 90:  # Very lenient threshold
                    name = classNames[id_].upper()
                    markAttendance(name)
                    color = (0, 255, 0)
                    status = "Recognized"
                else:
                    name = "Unknown"
                    color = (0, 0, 255)
                    status = "Not Recognized"
                
                # Draw name, confidence, and status
                cv2.rectangle(img, (x, y+h-35), (x+w, y+h), color, cv2.FILLED)
                cv2.putText(img, f"{name}", 
                           (x+6, y+h-20), cv2.FONT_HERSHEY_COMPLEX, 
                           0.7, (255, 255, 255), 2)
                cv2.putText(img, f"Conf: {int(confidence)}%", 
                           (x+6, y+h-6), cv2.FONT_HERSHEY_COMPLEX, 
                           0.7, (255, 255, 255), 2)
                
                # Print debug information
                print(f"Status: {status}, Name: {name}, Confidence: {int(confidence)}%")
                
            except Exception as e:
                print(f"Error processing face: {str(e)}")
                continue
    
    # Display FPS
    cv2.putText(img, f"FPS: {int(fps)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the frame
    cv2.imshow('Face Recognition Attendance System', img)
    
    # Break loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Program ended successfully")