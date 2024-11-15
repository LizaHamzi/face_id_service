import os
import numpy as np
import face_recognition
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

app = FastAPI()


SIGNATURES_PATH = 'Signatures.npy'

UPLOAD_DIR = r"C:\Users\aziz\OneDrive - Institut Teccart\Bureau\medical_project\uploaded_images"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)


if os.path.exists(SIGNATURES_PATH):
    signatures = np.load(SIGNATURES_PATH, allow_pickle=True)
else:
    signatures = np.empty((0, 129))  # 128 pour l'encodage et 1 pour l'email


@app.post("/add-signature/")
async def add_signature(file: UploadFile = File(...), email: str = Form(...)):
    try:
        img = face_recognition.load_image_file(file.file)
        encodings = face_recognition.face_encodings(img)

        if len(encodings) == 0:
            raise HTTPException(status_code=400, detail="Aucun visage détecté sur la photo.")

        encoding = encodings[0].tolist()
        encoding.append(email)  

        global signatures
        signatures = np.vstack([signatures, encoding])

        np.save(SIGNATURES_PATH, signatures)
        return {"message": f"Signature faciale ajoutée avec succès pour l'email : {email}."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération de la signature : {str(e)}")


@app.post("/verify-face-id/")
async def verify_face_id(email: str = Form(...), file: UploadFile = File(...)):
    print(f"email recu : {email}")
    try:
        img = face_recognition.load_image_file(file.file)
        encodesCurrent = face_recognition.face_encodings(img)

        if len(encodesCurrent) == 0:
            raise HTTPException(status_code=400, detail="Aucun visage détecté sur la photo.")

        global signatures
        if os.path.exists(SIGNATURES_PATH):
            signatures = np.load(SIGNATURES_PATH, allow_pickle=True)
        else:
            raise HTTPException(status_code=500, detail="Pas de signatures disponibles.")

        print(f"Nombre de signatures chargées : {len(signatures)}")

        for signature in signatures:
            stored_email = signature[-1]
            print(f"Comparaison avec l'utilisateur : {stored_email}")

            if stored_email == email:
                encoding = signature[:-1].astype('float')
                print(f"Encodage trouvé pour l'utilisateur : {encoding}")

                # Comparer avec l'encodage de l'image capturée
                matches = face_recognition.compare_faces([encoding], encodesCurrent[0])
                face_distance = face_recognition.face_distance([encoding], encodesCurrent[0])

                print(f"Distance faciale : {face_distance[0]}")

                if matches[0] and face_distance[0] < 0.6:
                    return {"message": "Double authentification réussie."}
                else:
                    raise HTTPException(status_code=401, detail=f"Authentification échouée : distance faciale trop élevée ({face_distance[0]}).")

        raise HTTPException(status_code=400, detail="Utilisateur non trouvé ou signature non correspondante.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la vérification : {str(e)}")
