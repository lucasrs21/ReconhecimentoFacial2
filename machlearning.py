from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os

# Coleta de arquivos necessarios e pasta de saida.
basedir = os.path.dirname(__file__)
output = basedir + "/output"

# Analise de dataset e criacao do embeddings.pickle. 
def extract_embeddings():
    dataset = basedir + "/dataset"
    embeddings = output + "/embeddings.pickle"
    embedding_model = basedir + "/openface_nn4.small2.v1.t7"
    face_model = basedir + "/face_detection_model"

    # Carrega o detector de rostos do disco (prototxt e caffemodel)
    protoPath = face_model + "/deploy.prototxt"
    modelPath = face_model + "/res10_300x300_ssd_iter_140000.caffemodel"
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    print("[INFO] Carregando face recognizer...")
    embedder = cv2.dnn.readNetFromTorch(embedding_model)

    # Adquire os diretorios das imagens de entrada do dataset
    print("[INFO] Quantificando rostos...")
    imagePaths = list(paths.list_images(dataset))

    # Inicializa as listas de embeddings de cada rosto e repectivos nomes das pessoas
    knownEmbeddings = []
    knownNames = []

    # Inicializa o numero total de rostos processados
    total = 0

    # Passa um loop pelos diretorios das imagens
    for (i, imagePath) in enumerate(imagePaths):
        # Extrai o nome do individuo da pasta do diretorio das imagens
        print("[INFO] Processando imagens {}/{}".format(i + 1,
            len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]

        # Carrega a imagem, redimensiona para uma largura de 600px (mantendo aspect ratio constante)
        # e coleta as dimensoes da imagem
        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]

        # Construcao de um blob da imagem
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # Aplicando detector de rosto do OpenCV (baseado em deep learning) para localizar rostos nas imagens
        detector.setInput(imageBlob)
        detections = detector.forward()

        # Verificando a deteccao de pelo menos um rosto
        if len(detections) > 0:
            # Estamos assumindo que cada imagem tenha apenas UM rosto,
            # entao colocamos um box na que tiver maior probabilidade
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]

            # Verificando que a deteccao para a maior probabilidade tambem significa
            # o teste de menor probabilidade (dessa forma, ajudando a filtrar deteccoes ruins)
            if confidence > 0.9:
                # Computar as coordenadas (x, y) do box para o rosto
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # extract the face ROI and grab the ROI dimensions
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # Verificar dimensoes de rosto suficientemente largas
                if fW < 20 or fH < 20:
                    continue

                # construct a blob for the face ROI, then pass the blob
                # through our face embedding model to obtain the 128-d
                # quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                    (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                # Adicao do nome da pessoa + embedding correspondente das suas respectivas listas
                knownNames.append(name)
                knownEmbeddings.append(vec.flatten())
                total += 1

    # Saida de embeddings + nomes
    print("[INFO] Serializando {} encodings...".format(total))
    data = {"embeddings": knownEmbeddings, "names": knownNames}
    while True:
        n = input("O arquivo embeddings já existe? [y/n] \n")
        if n == 'y' or n == 'Y':
            f = open(embeddings, "ab")
            f.write(pickle.dumps(data))
            f.close
            break
        elif n == 'n' or n == 'N':
            f = open(embeddings, "wb")
            f.write(pickle.dumps(data))
            f.close()
            break
        else:
            continue

# Funcao que treina a rede neural a partir do arquivo de embeddings &
# gera os arquivo recognizer e le (ambos são '.pickle').
def train_model():
    # Pegando os arquivos necessários ao programa
    embeddings = output + "/embeddings.pickle"
    label = output + "/le.pickle"
    recog = output + "/recognizer.pickle"

    # Carregando face embeddings
    print("[INFO] Carregando face embeddings...")
    data = pickle.loads(open(embeddings, "rb").read())

    # Codificando os labels
    print("[INFO] Codificando labels...")
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])

    # Treinando o modelo usado para aceitar embeddings do tipo 128-d do rosto
    # e, em seguida, criando o reconhecimento facial
    print("[INFO] Treinando modelo...")
    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)

    # Saida do reconhecimento facial (recognizer.pickle)
    f = open(recog, "wb")
    f.write(pickle.dumps(recognizer))
    f.close()

    # Saida do codificador de label (le.pickle)
    f = open(label, "wb")
    f.write(pickle.dumps(le))
    f.close()

extract_embeddings()
z = input('Iniciar Treinamento? [y/n] \n')

if z == 'y':
    train_model()
    print('[INFO] Procedimento encerrado...')
else:
    print('[INFO] Encerrando código...')