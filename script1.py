import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as matimg
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.datasets import mnist

'''
img = matimg.imread("cut3.png")
img_new = np.array([1] * (160**2)).reshape(160, 160)
img_new[:, 67: 93] = img
plt.imshow(img_new)
plt.show()
'''

'''
img2 = Image.open("new_cut3.png")
print(type(img2))
out = img2.resize((28, 28))
out.save("cut3_copy.png")
'''


model = tf.keras.models.load_model("model_save/model_0.h5")
img = matimg.imread("img_new_1.png")
print("img shape: ", img.shape)
img = np.array([img[:, :, 0]])
img_test = np.expand_dims(img, axis=-1)
res = model.predict(img_test)


def get_max_number(output_list):
    max_number = max(output_list)
    for i, num in enumerate(output_list):
        if num == max_number:
            return i


out = get_max_number(res[0])
print(res)

