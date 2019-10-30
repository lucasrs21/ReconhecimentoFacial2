import cv2
import numpy as np
import os
from cria_dir import Cria_dir

class Photos_by_video:

    def _init_(self):
        self.base_dir
        self.entrada
        self.x
        self.pasta
        self.vid

    def web_or_video(self):
        self.base_dir = os.path.dirname(__file__)
        self.entrada = input('web/video?')
        if self.entrada == 'video':
            self.x = input('Nome do vídeo: ')
            formato_video = input('Formato do vídeo: ')
            # set video file path of input video with name and extension
            self.vid = cv2.VideoCapture(self.base_dir + '/videos/' + self.x + '.' + formato_video)

        elif self.entrada == 'web':
            self.x = input('Nome da pessoa a ser cadastrada: ')
            # turn on the webcam
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
                print ('Creating...' + name)
                cv2.imwrite(name, frame)
                aux += 1
            # next frame
            index += 1