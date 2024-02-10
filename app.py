from flask import Flask, render_template, Response
import cv2
import imutils
from imutils import face_utils
import dlib
from scipy.spatial import distance
import time
from pygame import mixer
from threading import Thread, Lock
from twilio.rest import Client

app = Flask(__name__)

# Initialize Pygame mixer for audio alerts
mixer.init()
mixer.music.load("music.wav")

# Twilio Account SID and Auth Token
account_sid = "AC483ec5e771443b7879984483c75237f4"
auth_token = "a8758997c8a5d22c231bfea41b267d99"
client = Client(account_sid, auth_token)

# Function to send emergency message using Twilio
def send_emergency_message(emergency_number):
    message = client.messages.create(
        to=emergency_number,
        from_="+16592663439",
        body="Emergency! The user is showing signs of drowsiness for an extended period. Please check on them."
    )
    print(f"Emergency message sent to {emergency_number} - Message SID: {message.sid}")

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Threshold for blink detection
thresh = 0.25
# flag = 0
frame_check = 20  # Number of consecutive frames for which EAR must be below the threshold to trigger an alarm
alarm_duration = 2  # Duration (in seconds) to trigger the alarm

# Define the indices for left and right eye landmarks in the facial landmarks model
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['right_eye']

# Initialize face detector and facial landmark predictor
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Open a connection to the camera
cap = cv2.VideoCapture(0)

# Emergency number input from the user
emergency_number = "+919499174939"

# Flag to indicate when the camera is in use
camera_in_use = True

# Variables for closed eyes detection
closed_eyes_start_time = None
alarm_triggered = False

# Variable to store blink count
blink_count = 0

# Lock for synchronizing updates to blink_count
blink_count_lock = Lock()

# Function to generate frames for the webcam feed
def generate_frames():
    global alarm_triggered, closed_eyes_start_time, camera_in_use, blink_count

    while camera_in_use:
        # Capture a frame from the camera
        ret, frame = cap.read()

        # Resize the frame for better processing speed
        frame = imutils.resize(frame, width=450)

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        subjects = detect(gray, 0)

        # Loop over the detected faces
        for subject in subjects:
            # Predict facial landmarks
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)

            # Extract left and right eye coordinates
            left_eye = shape[lStart:lEnd]
            right_eye = shape[rStart:rEnd]

            # Calculate the Eye Aspect Ratio (EAR) for each eye
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            # Check for closed eyes
            if ear < thresh:
                # flag += 1
                if closed_eyes_start_time is None:
                    closed_eyes_start_time = time.time()
                else:
                    closed_eyes_duration = time.time() - closed_eyes_start_time

                    if closed_eyes_duration >= alarm_duration and not alarm_triggered:
                        # If closed eyes duration exceeds the threshold, trigger an alert
                        cv2.putText(frame, "***ALERT***", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(frame, "***ALERT***", (10, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        mixer.music.play()
                        send_emergency_message(emergency_number)

                        # Increment blink count using the lock
                        with blink_count_lock:
                            blink_count += 1

                        alarm_triggered = True


            else:
                # Reset the closed eyes timer if eyes are open
                closed_eyes_start_time = None
                alarm_triggered = False

        # Encode the frame as JPEG
        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

# Route for the index page
@app.route('/')
def index():
    # Provide blink_count to the template
    with blink_count_lock:
        return render_template('index.html', blink_count=blink_count)

# Route for streaming the webcam feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__': 
    Thread(target=app.run, args=('0.0.0.0', 5000)).start()
