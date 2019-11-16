import cv2
import numpy as np
import os
from PIL import Image, ImageEnhance
import shutil

def Cria_dir(y):
	#Define o diretório base
	base_dir = os.path.dirname(__file__)
	# Cria um novo diretório caso esse já não exista
	if not os.path.exists(base_dir + '/' + y):
		os.makedirs(base_dir + '/' + y)