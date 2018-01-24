import os
import imageio
import numpy as np
from PIL import Image
from datetime import datetime

SCALE = 286
CROP = 256


def log(file, str, end="\n"):
    print("%s: %s" % (datetime.now().strftime('%Y/%m/%d %X'), str), end=end)
    file.write("%s: %s%s" % (datetime.now().strftime('%Y/%m/%d %X'), str, end))
    file.flush()


def get_prefix():
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    prefix = 'run-{}/'.format(now)
    if not os.path.exists(prefix):
        os.mkdir(prefix)
    return prefix


def preprocess(image):
    return image / 127.5 - 1


def transfrom(image, off_w, off_h):
    image = image.resize((SCALE, SCALE), Image.BILINEAR)
    image = image.crop((off_w, off_h, off_w + CROP, off_h + CROP)).getdata()
    image = np.array(image).reshape((CROP, CROP, 3))
    image = preprocess(image)
    return image


def get_image(file_name):
    image = Image.open(file_name)
    width, height = image.size
    offset = np.random.randint(0, SCALE - CROP + 1, size=(2,))
    image_a = transfrom(image.crop((0, 0, width // 2, height)), offset[0], offset[1])
    image_b = transfrom(image.crop((width // 2, 0, width, height)), offset[0], offset[1])
    return [image_a, image_b]


def save_images(image, file_name):
    return imageio.imwrite(file_name, image)
