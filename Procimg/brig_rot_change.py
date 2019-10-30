from PIL import Image, ImageEnhance
import os
import shutil
from cria_dir import Cria_dir

class Brig_rot_change:

    def _init(self):
        # Definicao de diretorios de arquivos de entrada
        # e de parametros para brilho e rotacao
        self.base_dir
        self.bright
        self.rotate

    def Rotate(self):
        Cria_dir('/new_dataset')
        # Bloco responsavel pela mudanca de rotacao das imagens,
        # uma a uma, para os angulos definidos na lista rotate
        self.base_dir = os.path.dirname(__file__)
        self.rotate = [-15, -20, -25, -30, -35, 15, 20, 25, 30, 35]
        for dir in os.listdir(self.base_dir + '/updated_dataset'):
            if not os.path.exists(self.base_dir + '/new_dataset/' + dir):
                os.makedirs(self.base_dir + '/new_dataset/' + dir)
            for file in os.listdir(self.base_dir + '/updated_dataset/' + dir):
                for i in self.rotate:
                    img = Image.open(self.base_dir + '/updated_dataset/' + dir + '/' + file)
                    file_name, file_extension = os.path.splitext(file)
                    name = file_name + str(i) + file_extension
                    img.rotate(i, expand=True).save(self.base_dir + '/new_dataset/' + dir + '/' + name,'jpeg')

    def Bright(self):
        Cria_dir('/dataset')
        # Bloco responsavel pela mudanca de brilho das imagens,
        # uma a uma, para as porcentagens definidas na lista bright
        self.base_dir = os.path.dirname(__file__)
        self.bright = [0.4, 0.5, 0.6, 0.7]
        for dir in os.listdir(self.base_dir + '/new_dataset'):
            if not os.path.exists(self.base_dir + '/dataset/' + dir):
                os.makedirs(self.base_dir + '/dataset/' + dir)
            for file in os.listdir(self.base_dir + '/new_dataset/' + dir):
                for i in self.bright:
                    img = Image.open(self.base_dir + '/new_dataset/' + dir + '/' + file)
                    en = ImageEnhance.Brightness(img)
                    file_name, file_extension = os.path.splitext(file)
                    img = en.enhance(i)
                    name = file_name + '.' + str(i)
                    img.save(self.base_dir + '/dataset/' + dir + '/' + name,'jpeg')