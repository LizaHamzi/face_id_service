import os
import cv2
import numpy as np
import face_recognition


paths = [
    r"C:\Users\aziz\OneDrive - Institut Teccart\Bureau\medical_project\medical_application\media\image_doctors",
    r"C:\Users\aziz\OneDrive - Institut Teccart\Bureau\medical_project\medical_application\media\image_patients"
]
images_list = []  
emails = []


def get_email_from_filename(filename):
    name = filename.split('.')[0]
    parts = name.split('_', 2)
    if len(parts) >= 3:
        email = f'{parts[0]}@{parts[1]}.{parts[2]}'
    else:
        email = name
    return email


def load_images_from_folders(paths):
    for path in paths:
        myList = os.listdir(path)
        for img_name in myList:
            if img_name.lower().endswith(('.jpeg', '.jpg', '.png', '.bmp', '.tiff')):
                curImg = cv2.imread(f'{path}/{img_name}')
                images_list.append(curImg)
                
                
                email = get_email_from_filename(img_name)
                emails.append(email)

load_images_from_folders(paths)

# Détection des visages et extraction des caractéristiques
def extractFaceFeatures(Images, emails):
    features = []
    count = 1
    for image, email in zip(Images, emails):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        feature = face_recognition.face_encodings(img)
        if len(feature) > 0:
            feature = feature[0].tolist() + [email]
            features.append(feature)
            print(f'{int((count/(len(Images)))*100)} % extracted')
        count += 1
    array = np.array(features)
    np.save('Signatures.npy', array)
    print('Signatures saved!')
extractFaceFeatures(images_list, emails)
