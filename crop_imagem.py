# Import libraries
import os
import cv2
import numpy as np
from cria_dir import Cria_dir

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

			print("Extracted " + str(count) + " faces from all images")