import os
import glob
import dlib
import numpy as np
import _pickle as cPickle
from PIL import Image

# Detectores de face e pontos faciais
faceDetector = dlib.get_frontal_face_detector()
landmarkDetector = dlib.shape_predictor('../resources/shape_predictor_68_face_landmarks.dat')

# Usa redes neurais convolucionais
facialRecognition = dlib.face_recognition_model_v1('../resources/dlib_face_recognition_resnet_model_v1.dat')

labels = {}
i = 0
facialDescriptors = None

# Percorre as imagens de treinamento enquanto detecta faces
for arq in glob.glob(os.path.join('../yalefaces/training', "*.gif")):
    # Imagens do yalefaces Dataset são .gif portanto é necessário convertê-las
    imgFace = Image.open(arq).convert('RGB')
    img = np.array(imgFace, 'uint8')

    # Armazena bounding boxes das faces encontradas
    detectedFaces = faceDetector(img, 2)

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

        #labels para as caracteristicas
        labels[i] = arq
        i += 1

# Salva arquivos
np.save('../resources/descriptors.npy', facialDescriptors)
with open("../resources/labels.pickle", "wb") as f:
    cPickle.dump(labels, f)