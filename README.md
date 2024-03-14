# Drowsiness Detection System

This Python script utilizes facial landmarks detection to detect signs of drowsiness through webcam feed analysis. When drowsiness is detected, an alarm sounds and an emergency alert is sent via Twilio. 

## Dependencies

- Flask
- OpenCV
- imutils
- dlib
- scipy
- pygame
- twilio

## Usage

Clone the repository and run the Python script. Access the webcam feed and blink count via a web browser.

## Configuration

Configure Twilio credentials and emergency number in the script before running.

## Files

- `drowsiness_detection.py`: Python script
- `music.wav`: Alarm sound
- `shape_predictor_68_face_landmarks.dat`: Facial landmarks model
- `templates/index.html`: HTML template

## License

MIT License

