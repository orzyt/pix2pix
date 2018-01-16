from datetime import datetime

def log(file, str, end="\n"):
    print("%s: %s" % (datetime.now(), str), end=end)
    file.write("%s: %s%s" % (datetime.now(), str, end))
    file.flush()

def preprocess(img):
    # [0, 1] => [-1, 1]
    return img * 2 - 1

def deprocess(img):
    # [-1, 1] => [0, 1]
    return (img + 1) / 2