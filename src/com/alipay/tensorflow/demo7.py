# https://my.oschina.net/112612/blog/1594140
import numpy as np

from PIL import Image

x = np.zeros([64, 40 * 110])

image = Image.open('/Users/yejiadong/dev/python/intelligent/src/flowers/BW2I.png').convert('L')
image = np.array(image)

x[1,:] = 1 * image.flatten()


print(x)


