import cv2
import numpy as np
import os
from PIL import Image, ImageEnhance
import shutil
from procimagelib import Brig_rot_change, Crop_imagem, Photos_by_video


captura = Photos_by_video()
captura.web_or_video()
captura.extrai_frames()

cropagem = Crop_imagem()
cropagem.define_path()
cropagem.crop_img()

var = Brig_rot_change()
var.Rotate()
var.Bright()