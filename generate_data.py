from datetime import datetime as dt
from PIL import Image
import face_recognition
import cv2
import os

DIR_PATH = os.path.abspath(os.path.dirname(__file__))

video_capture = cv2.VideoCapture(0)
face_locations = []
process_this_frame = True

training_path = 'model/data/training'
type = 'mask' # face or mask
face_size = (125, 125)
counter = 1

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left) in face_locations:
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        faceImage = frame[top:bottom, left:right]
        
        Image.fromarray(cv2.resize(faceImage, face_size)).save(os.path.join(DIR_PATH, training_path, type, f'{dt.now().isoformat()}_{counter}.png'))
        counter = counter + 1

        color = (0, 255, 0)

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom),
                      (right, bottom), color, cv2.FILLED)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
