from PIL import Image


def initTable():
    table = []
    for i in range(256):
        if i > 0 and i < 10:
            table.append(1)
        else:
            table.append(0)
    return table


for i in range(1, 5):
    im = Image.open("/Users/yejiadong/dev/python/intelligent/src/flowers/" + str(i) + ".jpeg")
    im = im.convert('L')
    # im.point(initTable(), '1')
    region = (17, 10, 90, 30)
    img = im.crop(region)
    img.show()
