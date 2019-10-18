from PIL import Image, ImageEnhance
import os
import shutil

base_dir = os.path.dirname(__file__)
bright = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for file in os.listdir(base_dir + 'in'):
    for i in bright:
        img = Image.open('in/' + file)
        en = ImageEnhance.Brightness(img)
        file_name, file_extension = os.path.splitext(file)
        img = en.enhance(i)
        name = file_name + str(i)
        img.save(name,'jpeg')
        shutil.move(base_dir + name, base_dir + 'out')