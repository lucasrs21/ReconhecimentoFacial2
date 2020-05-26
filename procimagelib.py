# Import libraries
import cv2
import numpy as np
import os
from PIL import Image, ImageEnhance
import shutil

def Cria_dir(y):
    # Funcao responsavel pela criacao 
    # de diretorios baseados em um diretorio base (base_dir)
	# Define o diretório base
	base_dir = os.path.dirname(__file__)
	# Cria um novo diretório caso esse já não exista
	if not os.path.exists(base_dir + '/' + y):
		os.makedirs(base_dir + '/' + y)

class Crop_imagem:

	def _init_(self):
		self.base_dir
		self.prototxt_path
		self.caffemodel_path
		self.model

	def define_path(self):
		#Define os 'paths'
		Cria_dir('/updated_dataset')
		self.base_dir = os.path.dirname(__file__)
		self.prototxt_path = os.path.join(self.base_dir + '/model_data/deploy.prototxt')
		self.caffemodel_path = os.path.join(self.base_dir + '/model_data/weights.caffemodel')

	def crop_img(self):
		# Read the model
		self.model = cv2.dnn.readNetFromCaffe(self.prototxt_path, self.caffemodel_path)
		# Loop through all images and strip out faces
		count = 0
		for dir in os.listdir(self.base_dir + '/initial_dataset'):
			Cria_dir('updated_dataset/' + dir)
			for file in os.listdir(self.base_dir + '/initial_dataset/' + dir):
				file_name, file_extension = os.path.splitext(file)
				if (file_extension in ['.png','.jpg']):
					print("Image path: {}".format(self.base_dir + '/initial_dataset/' + dir + '/' + file))
					image = cv2.imread(self.base_dir + '/initial_dataset/' + dir + '/' + file)
					(h, w) = image.shape[:2]
					blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

					self.model.setInput(blob)
					detections = self.model.forward()

					# Identify each face
					for i in range(0, detections.shape[2]):
						box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
						(startX, startY, endX, endY) = box.astype("int")

						confidence = detections[0, 0, i, 2]

						# If confidence > 0.5, save it as a separate file
						if (confidence > 0.5):
							count += 1
							frame = image[startY-90:endY+90, startX-90:endX+90]
							cv2.imwrite(self.base_dir + '/updated_dataset/' + dir + '/' + str(i) + '_' + file, frame)

			print("Foram extraídas " + str(count) + " faces de todas as imagens.")

class Brig_rot_change:

    def _init(self):
        # Definicao de diretorios de arquivos de entrada
        # e de parametros para brilho e rotacao
        self.base_dir
        self.bright
        self.rotate

    def Rotate(self):
        #Bloco responsavel pela mudanca de rotacao das imagens, 
        # uma a uma, para os angulos definidos na lista rotate.
        Cria_dir('new_dataset')
        self.base_dir = os.path.dirname(__file__)
        self.rotate = [-15, -20, -25, -30, -35, 0, 15, 20, 25, 30, 35]
        for dir in os.listdir(self.base_dir + '/updated_dataset'):
            Cria_dir('new_dataset/' + dir)
            for file in os.listdir(self.base_dir + '/updated_dataset/' + dir):
                for i in self.rotate:
                    try:
                        img = Image.open(self.base_dir + '/updated_dataset/' + dir + '/' + file)
                        file_name, file_extension = os.path.splitext(file)
                        name = file_name + str(i) + file_extension
                        img.rotate(i, expand=True).save(self.base_dir + '/new_dataset/' + dir + '/' + name,'jpeg')
                    except (OSError, FileNotFoundError):
                        os.remove(self.base_dir + '/updated_dataset/' + dir + '/' + file)
                        print('Frame {} foi deletado'.format(file))
                        break

    def Bright(self):
        #Bloco responsavel pela mudanca de brilho das imagens,
        # uma a uma, para as porcentagens definidas na lista bright
        Cria_dir('/dataset')
        self.base_dir = os.path.dirname(__file__)
        self.bright = [0.4, 0.5, 0.6, 0.7, 1.4, 1.5, 1.6, 1.7]
        for dir in os.listdir(self.base_dir + '/new_dataset'):
            Cria_dir('dataset/' + dir)
            for file in os.listdir(self.base_dir + '/new_dataset/' + dir):
                for i in self.bright:
                    img = Image.open(self.base_dir + '/new_dataset/' + dir + '/' + file)
                    en = ImageEnhance.Brightness(img)
                    file_name, file_extension = os.path.splitext(file)
                    img = en.enhance(i)
                    name = file_name + '.' + str(i) + file_extension
                    img.save(self.base_dir + '/dataset/' + dir + '/' + name,'jpeg')

class Photos_by_video:

    def _init_(self):
        self.base_dir
        self.entrada
        self.x
        self.pasta
        self.vid

    def web_or_video(self):
        self.base_dir = os.path.dirname(__file__)
        self.entrada = input('Selecione o método de captura [web/video]: ')
        if self.entrada == 'video':
            self.x = input('Nome do vídeo: ')
            formato_video = input('Formato do vídeo: ')
            # set video file path of input video with name and extension
            self.vid = cv2.VideoCapture(self.base_dir + '/videos/' + self.x + '.' + formato_video)

        elif self.entrada == 'web':
            self.x = input('Nome da pessoa a ser cadastrada: ')
            self.vid = cv2.VideoCapture(0)
        
        else:
            print('Resposta não válida')


    def extrai_frames(self):
        Cria_dir('/initial_dataset/' + self.x)
        #for frame identity
        fps = 60
        index = 0
        aux = 1
        while(True):
            # Extract images
            ret, frame = self.vid.read()
    
            if self.entrada == 'web':  
                #show the output frame
                cv2.imshow('Frame', frame)
                key = cv2.waitKey(1) & 0xFF
        
                # keyboard 'q' closes/stop
                if key == ord('q'):
                    break
    
            # end of frames
            if not ret: 
                break
            # Saves images
            if index%fps == 0:
                index = aux
                name = self.base_dir + '/initial_dataset/' + self.x + '/frame' + str(index) + '.jpg'      
                print ('Criando...' + name)
                cv2.imwrite(name, frame)
                aux += 1
            # next frame
            index += 1