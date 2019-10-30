import cv2
import numpy as np
import os
from PIL import Image, ImageEnhance
import shutil
from photos_by_video import Photos_by_video
from crop_imagem import Crop_imagem
from brig_rot_change import Brig_rot_change

captura = Photos_by_video()
captura.web_or_video()
captura.extrai_frames()

cropagem = Crop_imagem()
cropagem.define_path()
cropagem.crop_img()

var = Brig_rot_change()
var.Rotate()
var.Bright()