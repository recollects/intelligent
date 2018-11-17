import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

x = np.zeros([64, 114 * 450])

image = Image.open('/Users/yejiadong/dev/python/intelligent/src/flowers/BW2I.png').convert('L')
image.save('/Users/yejiadong/dev/python/intelligent/src/flowers/BW2I0.png')
img=mpimg.imread('/Users/yejiadong/dev/python/intelligent/src/flowers/BW2I0.png')
gray = rgb2gray(img)
img = np.array(gray)

x[1,:] = 1 * img.flatten()
# image.flatten('A')


print(x)


