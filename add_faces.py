import cv2
import pickle
import numpy as np
import os

# Create 'data' directory if it does not exist
if not os.path.exists('data/'):
    os.makedirs('data/')

# Initialize 'names.pkl' if not present (stores registered Aadhar numbers)
names_file = 'data/names.pkl'
if not os.path.exists(names_file):
    with open(names_file, 'wb') as f:
        pickle.dump([], f)  # Create an empty list for names if the file doesn't exist

# Initialize 'faces_data.pkl' if not present (stores face data)
faces_file = 'data/faces_data.pkl'
if not os.path.exists(faces_file):
    with open(faces_file, 'wb') as f:
        pickle.dump(np.empty((0, 7500)), f)  # Empty array with shape (0, 7500) to ensure consistency

# Load existing registered names from 'names.pkl'
with open(names_file, 'rb') as f:
    names = pickle.load(f)

# Load existing face data from 'faces_data.pkl'
with open(faces_file, 'rb') as f:
    faces = pickle.load(f)

# Initialize webcam capture
video = cv2.VideoCapture(0)

# Load pre-trained face detection model
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# List to store the face images of the new voter
faces_data = []

# Ask the user to enter their Aadhar number
name = input('Enter your Aadhar number: ')

# Number of frames to capture and process for each voter
framesTotal = 51  # Total number of face images to collect
captureAfter = 2  # Capture every 2nd frame to avoid duplicates
i = 0  # Frame counter

# Start capturing frames from the webcam
while True:
    ret, frame = video.read()  # Read a frame from the webcam
    if not ret:  # If the frame is not captured, show an error
        print("Error: Failed to capture frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale for better face detection
    faces_detected = facedetect.detectMultiScale(gray, 1.3, 5)  # Detect faces in the frame

    for (x, y, w, h) in faces_detected:
        crop_img = frame[y:y + h, x:x + w]  # Crop the detected face from the frame
        resized_img = cv2.resize(crop_img, (50, 50))  # Resize the face image to 50x50 pixels
        resized_img = resized_img.flatten()  # Flatten the 50x50 image into a 1D array of 7500 values

        # Store the face data only if we haven't reached the required number of images
        if len(faces_data) < framesTotal and i % captureAfter == 0:
            faces_data.append(resized_img)

        i += 1  # Increment frame counter
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)  # Display number of images captured
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw a rectangle around the detected face

    cv2.imshow('frame', frame)  # Display the frame with the face detection
    k = cv2.waitKey(1)  # Wait for key press

    # Stop capturing if 'q' is pressed or enough face images have been collected
    if k == ord('q') or len(faces_data) >= framesTotal:
        break

# Release the webcam and close any OpenCV windows
video.release()
cv2.destroyAllWindows()

# Convert collected face images list to numpy array
faces_data = np.asarray(faces_data)

# Append new Aadhar number to 'names.pkl' file
names.extend([name] * framesTotal)  # Store the name for each collected image
with open(names_file, 'wb') as f:
    pickle.dump(names, f)  # Save updated names list

# Append new face data to 'faces_data.pkl' file
faces = np.append(faces, faces_data, axis=0)  # Merge new face data with existing data
with open(faces_file, 'wb') as f:
    pickle.dump(faces, f)  # Save updated face dataset

print("Voter registered successfully!")  # Confirmation message
