import cv2
import numpy as np
import face_recognition


imgMessi = face_recognition.load_image_file("Faces/Messi.jpg")
imgRonaldo = face_recognition.load_image_file("Faces/Ronaldo-test.jpg")
imgMessi = cv2.cvtColor(imgMessi, cv2.COLOR_BGR2RGB)
imgRonaldo = cv2.cvtColor(imgRonaldo, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgMessi)[0]
encodeMessi = face_recognition.face_encodings(imgMessi)[0]

cv2.rectangle(imgMessi, (faceLoc[3], faceLoc[0]),
              (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

print(encodeMessi)

faceLocTest = face_recognition.face_locations(imgRonaldo)[0]
encodeRonaldo = face_recognition.face_encodings(imgRonaldo)[0]

cv2.rectangle(imgRonaldo, (faceLocTest[3], faceLocTest[0]),
              (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)
print(encodeRonaldo)

results = face_recognition.compare_faces([encodeMessi], encodeRonaldo)
print(results)

cv2.imshow('Messi', imgMessi)
cv2.imshow("Messi Test", imgRonaldo)
cv2.waitKey(0)
