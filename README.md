# Drowsiness Detection System

This Python script utilizes facial landmarks detection to detect signs of drowsiness through webcam feed analysis. When drowsiness is detected, an alarm sounds and an emergency alert is sent via Twilio. 

The project aims to enhance road safety by reducing accidents caused by driver fatigue. The solution uses computer vision and machine learning to monitor driver behavior through a camera, detecting signs of drowsiness such as eyelid movement, yawning, and head position. When drowsiness is detected, the system sends real-time alerts to prompt the driver to take a break and also sends an emergency message to the number initially entered by the driver. This technology aims to significantly lower the number of fatigue-related accidents, potentially saving lives and reducing injuries on the road.

## Dependencies

- Flask
- OpenCV
- imutils
- dlib
- scipy
- pygame
- Twilio

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



![IMG-20240604-WA0006](https://github.com/VanshitaKhera/Drowsiness-Detection/assets/154512400/98ebe91c-9d57-412a-a415-b4b1f0885de0)

![IMG-20240604-WA0007](https://github.com/VanshitaKhera/Drowsiness-Detection/assets/154512400/4063e8db-6b19-47a1-939f-0ee59999a68e)


![IMG-20240604-WA0008](https://github.com/VanshitaKhera/Drowsiness-Detection/assets/154512400/57142687-cef2-4177-84f2-25326a9091de)

![WhatsApp Image 2024-06-04 at 14 19 40_46a7017d](https://github.com/VanshitaKhera/Drowsiness-Detection/assets/154512400/79342fb9-f721-4016-a9c6-0f6b1afefa2a)

