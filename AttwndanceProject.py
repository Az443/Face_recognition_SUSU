from datetime import datetime
import cv2
import numpy as np
import face_recognition 
import os



pathP = 'imagesAtt'
img_ = []
class_The_Names = []
TheList = os.listdir(pathP)
#print(TheList)  # this will print the photos names with suffixes 

for cla in TheList:
    cur_Img = cv2.imread(f'{pathP}/{cla}')
    img_.append(cur_Img)
    class_The_Names.append(os.path.splitext(cla)[0])
print(class_The_Names)    # this will print photos names without suffixes

def findEncodings(img_):
    List_Of_Encode = []
    for img_O in img_:
        img_O = cv2.cvtColor(img_O, cv2.COLOR_BGR2RGB)
        encoded_ph = face_recognition.face_encodings(img_O)[0]
        List_Of_Encode.append(encoded_ph)
    return List_Of_Encode

def mark_The_Attendance(name):
    with open('Att_List.csv', 'r+') as S:
        List_Of_Data = S.readlines()
        Listed_Names =[]
        for line in List_Of_Data:
            arrivals = line.split(',')
            Listed_Names.append(arrivals[0])
        if name not in Listed_Names:
            now = datetime.now()
            date_string = now.strftime('%H:%M')
            S.writelines(f'\n{name}, {date_string}')


ListOfEncode_KnownFaces = findEncodings(img_)
print(len(ListOfEncode_KnownFaces)) 


Capture = cv2.VideoCapture(0)

while True:
    success, img = Capture.read()
    Ti_Img = cv2.resize(img, (0,0), None, 0.25, 0.25)
    Ti_Img = cv2.cvtColor(Ti_Img, cv2.COLOR_BGR2RGB)

    Face_Cur_Frame = face_recognition.face_locations(Ti_Img)
    encode_Cur_Frame = face_recognition.face_encodings(Ti_Img, Face_Cur_Frame)

    for encodeFace, faceLoc in zip(encode_Cur_Frame, Face_Cur_Frame):
        The_matches = face_recognition.compare_faces(ListOfEncode_KnownFaces, encodeFace)
        faceDis = face_recognition.face_distance(ListOfEncode_KnownFaces, encodeFace)
        print(faceDis)
        match_Index = np.argmin(faceDis)

        if The_matches[match_Index]:
            name_Char = class_The_Names[match_Index].upper()
            print(name_Char)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img, (x1,y1),(x2,y2), (0,255,0), 2)
            cv2.rectangle(img, (x1,y2-35), (x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(img, name_Char, (x1+6,y2-6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
            mark_The_Attendance(name_Char)


    cv2.imshow('webcam', img)
    cv2.waitKey(1)
