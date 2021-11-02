from tensorflow.keras.models import load_model
import tensorflow as tf
import face_recognition
import cv2
import numpy as np
import os

# Code Source: https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam_faster.py

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

known_face_encodings = []
known_face_names = []

model_path = 'model/my_model'
model = load_model(model_path)
face_size = (125, 125)

dir = '../../FFHS_Bilder/'

if os.path.exists(dir):
    for filename in os.listdir(dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            print(filename)
            print(os.path.join(dir, filename))
            img = face_recognition.load_image_file(os.path.join(dir, filename))
            img_encoding = face_recognition.face_encodings(img)[0]
            # Create arrays of known face encodings and their names
            known_face_encodings.append(img_encoding)
            known_face_names.append(filename.split('.')[0])


# Initialize some variables
face_locations = []
face_names = []
process_this_frame = True

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
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:

            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            if len(known_face_encodings) > 0:
                face_distances = face_recognition.face_distance(
                    known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        faceImage = frame[top:bottom, left:right]

        image = tf.expand_dims(cv2.resize(faceImage, face_size), 0)
        result = model.predict(image)
        prediction = 'mask' if result[0][0] > result[0][1] else 'face'
        # Choose color: green means ok and red means not good...
        if prediction == 'mask':
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35),
                      (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    font, 0.7, (0, 0, 0), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
