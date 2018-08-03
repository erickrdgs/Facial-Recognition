import os
import glob
import dlib
import cv2
import numpy as np

# Detectores de face e pontos faciais
faceDetector = dlib.get_frontal_face_detector()
landmarkDetector = dlib.shape_predictor('../resources/shape_predictor_68_face_landmarks.dat')

# Usa redes neurais convolucionais
facialRecognition = dlib.face_recognition_model_v1('../resources/dlib_face_recognition_resnet_model_v1.dat')

# Carrega descritores faciais (treinamento)
facialDescriptors = np.load("../resources/descriptors.npy")

# Percorre as imagens de validação enquanto detecta faces
for arq in glob.glob(os.path.join('../photos/validation', "*.jpg")):
    img = cv2.imread(arq)

    detectedFaces = faceDetector(img, 2)

    # Aplica reconhecimento facial para cada bounding box detectada na imagem
    for face in detectedFaces:
        l, t, r, b = (int(face.left()), int(face.top()), int(face.right()), int(face.bottom()))

        # Gera descritores faciais
        landmarks = landmarkDetector(img, face)
        facialDescriptor = facialRecognition.compute_face_descriptor(img, landmarks)

        listDescriptor = [df for df in facialDescriptor]
        npArray = np.asarray(listDescriptor, dtype=np.float64)
        npArray = npArray[np.newaxis, :]

        # Cálculo da distância euclidiana (semelhança)
        dists = np.linalg.norm(npArray - facialDescriptors, axis=1)
        min = np.argmin(dists)

        # Limiar
        if dists[min] <= 0.5:
            cv2.rectangle(img, (l, t), (r, b), (0, 255, 255), 2)
            cv2.putText(img, "Obama", (r, t), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))

    cv2.imshow("Face Recognition", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()