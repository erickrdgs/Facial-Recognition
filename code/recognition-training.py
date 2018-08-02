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

i = 0
facialDescriptors = None

# Percorre as imagens de treinamento enquanto detecta faces
for arq in glob.glob(os.path.join('../photos/training', "*.jpg")):
    img = cv2.imread(arq)

    # Armazena bounding boxes das faces encontradas
    detectedFaces = faceDetector(img, 1)

    # Extrai pontos faciais nas bounding boxes
    for face in detectedFaces:
        landmarks = landmarkDetector(img, face)

        # Descreve a face encontrada com 128 caracteristicas principais (mais importantes)
        facialDescriptor = facialRecognition.compute_face_descriptor(img, landmarks)

        # Cria uma lista com as caracteristicas e converte para um numpy
        listdescriptors = [fd for fd in facialDescriptor]
        npArray = np.asarray(listdescriptors, dtype=np.float64)
        npArray = npArray[np.newaxis, :]

        # Formatação
        if facialDescriptors is None:
            facialDescriptors = npArray
        else:
            facialDescriptors = np.concatenate((facialDescriptors, npArray), axis=0)

# Salva arquivos
np.save('../resources/descriptors.npy', facialDescriptors)
