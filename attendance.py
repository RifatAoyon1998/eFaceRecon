import numpy as np
import face_recognition as fr
import cv2
import datetime
import pygame
import pyttsx3

speaker = pyttsx3.init()
video_capture = cv2.VideoCapture(0)

image = fr.load_image_file("1671156.jpg")
face_encoding = fr.face_encodings(image)[0]
image2 = fr.load_image_file("mark.jpg")
face_encoding2 = fr.face_encodings(image2)[0]
image3 = fr.load_image_file("elon.jpeg")
face_encoding3 = fr.face_encodings(image3)[0]
known_face_encondings = [face_encoding,face_encoding2,face_encoding3]
known_face_names = ["Aoyon","Mark","Elon"]
fn=""
nameList=[]
def mark_attendance(name):

    with open('atnd.csv', 'a') as f:
        if name not in nameList:
            nameList.append(name)
            date_time_string = datetime.datetime.now().strftime("%y/%m/%d %H:%M:%S")
            f.writelines(f'\n{name},{date_time_string}')



while True:
    ret, frame = video_capture.read()

    rgb_frame = frame[:, :, ::-1]

    face_locations = fr.face_locations(rgb_frame)
    face_encodings = fr.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        matches = fr.compare_faces(known_face_encondings, face_encoding)

        name = "Unknown"

        face_distances = fr.face_distance(known_face_encondings, face_encoding)

        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            mark_attendance(name)
        fn=name
        if name=="Unknown":
            pygame.init()
            pygame.mixer.music.load("alert.wav")
            pygame.mixer.music.play()
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Webcam_facerecognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()

cv2.destroyAllWindows()