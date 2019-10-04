import cv2
import numpy as np
import os

# set video file path of input video with name and extension
vid = cv2.VideoCapture('/home/matheusabr/Vídeos/matheus.webm')


#if not os.path.exists('images'):
#    os.makedirs('images')

#for frame identity
index = 0
aux = 1
while(True):
    # Extract images
    ret, frame = vid.read()
    # end of frames
    if not ret: 
        break
    # Saves images
    if index%10 == 0:
        index = aux
        name = '/home/matheusabr/Imagens/frames/frame' + str(index) + '.jpg'
        print ('Creating...' + name)
        cv2.imwrite(name, frame)
        aux += 1

    # next frame
    index += 1