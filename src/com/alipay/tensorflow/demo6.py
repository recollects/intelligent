import numpy as np
from numpy.ctypeslib import types

MAX_CAPTCHA = 4
CHAR_SET_LEN = 36

# def name2vec(name):
#     vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
# for i, c in enumerate(name):
#     idx = i * CHAR_SET_LEN + ord(c) - 65
#     vector[idx] = 1
# print("name2vec:" + vector)
# return vector

# def vec2name(vec):
#     name = []
#     for i in vec:
#         a = chr(i + 97)
#         name.append(a)
#     print("vec2name:"+name)
#     return "".join(name)


vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
for i, c in enumerate("94F2"):
    if type(c) is int:
        idx = i * 36 + ord(c) - 48
        # print(idx)
    else:
        idx = i * 36 + ord(c) - 65
        # print(idx)
    vector[idx] = 1
print(vector)

name = []
j = 0
for i in vector:
    j + 1;
    print(i)
    # if j > 26:
    #     a = chr(i + 48)
    # else:
    #     a = chr(i + 65)

print("vec2name:" + name)
str = "".join(name)
print(str)
