from PIL import Image, ImageEnhance
import os
import shutil

# Definicao de diretorios de arquivos de entrada
# e de parametros para brilho e rotacao
base_dir = os.path.dirname(__file__)
bright = [0.4, 0.5, 0.6, 0.7]
rotate = [-15, -20, -25, -30, -35, 15, 20, 25, 30, 35]

# Bloco responsavel pela mudanca de rotacao das imagens,
# uma a uma, para os angulos definidos na lista rotate
for file in os.listdir(base_dir + '/dataset/matheus'):
    for i in rotate:
        img = Image.open(base_dir + '/dataset/matheus/' + file)
        file_name, file_extension = os.path.splitext(file)
        name = file_name + str(i) + file_extension
        img.rotate(i, expand=True).save(name,'jpeg')
        shutil.move(base_dir + '/' + name, base_dir + '/in')

# Bloco responsavel pela mudanca de brilho das imagens,
# uma a uma, para as porcentagens definidas na lista bright
for file in os.listdir(base_dir + '/in'):
    for i in bright:
        img = Image.open(base_dir + '/in/' + file)
        en = ImageEnhance.Brightness(img)
        file_name, file_extension = os.path.splitext(file)
        img = en.enhance(i)
        name = file_name + '.' + str(i)
        img.save(name,'jpeg')
        shutil.move(base_dir + '/' + name, base_dir + '/out')