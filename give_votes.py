from sklearn.neighbors import KNeighborsClassifier  # Import KNN classifier
import cv2  # OpenCV for image processing
import pickle  # To load and save face data
import numpy as np  # For numerical operations
import os  # To check file paths and create directories
import csv  # For handling vote records
import time  # For timestamping votes
from datetime import datetime  # To format date and time
from win32com.client import Dispatch  # For text-to-speech (TTS)
import tkinter as tk  # Tkinter for GUI
from tkinter import Label, Button, StringVar, OptionMenu, messagebox  # GUI components
from PIL import Image, ImageTk  # To display the webcam feed in Tkinter GUI

# Function for Text-to-Speech (TTS)
def speak(str1):
    """Uses Windows Speech API to convert text to speech."""
    speaker = Dispatch("SAPI.SpVoice")
    speaker.Speak(str1)

# Load Face Data
if not os.path.exists('data/'):  
    os.makedirs('data/')  # Create 'data' directory if it does not exist

# Load stored names (voters' IDs) from pickle file
with open('data/names.pkl', 'rb') as f:
    LABELS = pickle.load(f)

# Load stored face encodings from pickle file
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

# Train the KNN classifier on the loaded face data
knn = KNeighborsClassifier(n_neighbors=5)  # Use 5 nearest neighbors
knn.fit(FACES, LABELS)  # Train the model using loaded face data and corresponding labels

# Initialize Webcam
video = cv2.VideoCapture(0)  # Start video capture using the default camera
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # Load pre-trained Haarcascade model for face detection

# CSV File Setup (For storing vote records)
CSV_FILE = "Votes.csv"  # File to store voting records
COL_NAMES = ["NAME", "PARTY", "DATE", "TIME"]  # Column names for CSV
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COL_NAMES)  # Create the CSV file with column headers if it doesn't exist

# GUI Setup
root = tk.Tk()  # Initialize main Tkinter window
root.title("Face Recognition Voting System")  # Set window title
root.geometry("700x550")  # Set window size

# Label for the title
label = Label(root, text="Face Recognition Voting System", font=("Arial", 16))
label.pack(pady=10)

# Label for displaying the camera feed
camera_label = Label(root)
camera_label.pack()

# Label to display recognized voter's name
name_label = Label(root, text="Recognized Name: ", font=("Arial", 14))
name_label.pack()

# Label to show voting status
vote_status = Label(root, text="", font=("Arial", 14))
vote_status.pack()

# Party selection dropdown
parties = ["BRS", "BJP", "YSRCP", "CONGRES", "NOTA"]  # List of available parties
selected_party = StringVar()  # Variable to store selected party
selected_party.set(parties[0])  # Set default selection to first party

party_label = Label(root, text="Select Your Party:", font=("Arial", 12))
party_label.pack(pady=5)

party_dropdown = OptionMenu(root, selected_party, *parties)  # Dropdown menu for party selection
party_dropdown.pack(pady=5)

# Function to update camera feed in Tkinter GUI
def update_camera():
    """Captures frames from the webcam and displays them in the GUI."""
    ret, frame = video.read()  # Capture a frame from the webcam
    if ret:
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally for a natural display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB format
        img = Image.fromarray(frame_rgb)  # Convert NumPy array to PIL Image
        img_tk = ImageTk.PhotoImage(image=img)  # Convert PIL Image to Tkinter-compatible format
        camera_label.img_tk = img_tk  # Keep a reference to avoid garbage collection issues
        camera_label.config(image=img_tk)  # Update the GUI label with the new image
    root.after(10, update_camera)  # Refresh the camera feed every 10ms

# Function to recognize face and vote
def recognize_and_vote():
    """Recognizes the voter's face and records their vote."""
    ret, frame = video.read()  # Capture a frame from the webcam
    if not ret:
        vote_status.config(text="Camera error. Try again.", fg="red")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    faces = facedetect.detectMultiScale(gray, 1.3, 5)  # Detect faces in the frame

    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w]  # Crop the detected face
        resized_img = cv2.resize(crop_img, (50, 50)).reshape(1, -1)  # Resize and flatten the image

        try:
            recognized_name = knn.predict(resized_img)[0]  # Predict the person's name using KNN
        except:
            recognized_name = "Unknown"  # If the prediction fails, mark as Unknown

        name_label.config(text=f"Recognized Name: {recognized_name}")  # Update the GUI with the recognized name

        if recognized_name != "Unknown":
            party = selected_party.get()  # Get selected party from dropdown
            ts = time.time()  # Get current timestamp
            date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")  # Format date
            timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")  # Format time

            # Save vote details to CSV file
            with open(CSV_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([recognized_name, party, date, timestamp])

            vote_status.config(text=f"Vote Cast for {party}!", fg="green")  # Update status message
            speak(f"Vote successfully cast for {party}")  # Announce the vote using TTS

            # Show a message box confirming the vote
            messagebox.showinfo("Vote Submitted", f"{recognized_name}, your vote for {party} has been recorded.")

    root.after(5000, lambda: vote_status.config(text=""))  # Clear vote status message after 5 seconds

# Start Camera Feed
update_camera()  # Start updating the camera feed in the GUI

# Add Voting Button
vote_button = Button(root, text="Scan & Vote", font=("Arial", 14), command=recognize_and_vote)
vote_button.pack(pady=20)

# Run the Tkinter event loop
root.mainloop()

# Release resources when GUI is closed
video.release()  # Release the camera
cv2.destroyAllWindows()  # Close any OpenCV windows
 