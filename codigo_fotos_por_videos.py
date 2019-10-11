import cv2
import numpy as np
import os

base_dir = os.path.dirname(__file__)
print(base_dir)

entrada = input('video/web?: ')

if entrada == 'video':
    x = input('Nome do vídeo: ')
    formato_video = input('Formato do vídeo: ')
    # set video file path of input video with name and extension
    vid = cv2.VideoCapture(base_dir + '/videos/' + x + '.' + formato_video)

if entrada == 'web':
    x = input('Nome da pessoa a ser cadastrada: ')
    # turn on the webcam
    vid = cv2.VideoCapture(0)

fps = 10

pasta = base_dir + '/dataset/' + x
if not os.path.exists(pasta):
    os.makedirs(pasta)

#for frame identity
index = 0
aux = 1
while(True):
    # Extract images
    ret, frame = vid.read()
    
    if entrada == 'web':  
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
        name = pasta + '/frame' + str(index) + '.jpg'      
        print ('Creating...' + name)
        cv2.imwrite(name, frame)
        aux += 1
    # next frame
    index += 1