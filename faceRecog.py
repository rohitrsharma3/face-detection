import cv2
import numpy as np
import face_recognition
import os

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


path = 'F:/face-detection/project/Faces'
images = []
names = []
myList = os.listdir(path)
print(myList)
for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    names.append(os.path.splitext(cls)[0])
print(names)

encodeListKnown = findEncodings(images)
print('Encoding complete for ' + str(len(encodeListKnown))+ ' images')

capture = cv2.VideoCapture(0)

while True:
    isTrue, frame = capture.read()
    frame_small = cv2.resize(frame,(0,0),None,0.25,0.25)
    frame_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(frame_small)
    encodeCurFrame = face_recognition.face_encodings(frame_small,facesCurFrame)

    for encodeFace,faceloc in zip(encodeCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDist = face_recognition.face_distance(encodeListKnown, encodeFace)
        closestMatch = np.argmin(faceDist)
        print(faceDist[closestMatch])

        if matches[closestMatch]:
            name = names[closestMatch].upper()
            print(name)
            y1,x2,y2,x1 = faceloc
            y1, x2, y2, x1 = y1*4 ,x2*4, y2*4, x1*4
            cv2.rectangle(frame,(x1,y1),(x2,y2), (0,255,0),2)
            cv2.rectangle(frame,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(frame,name,(x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)


    cv2.imshow('webcame', frame)
    cv2.waitKey(1)
