import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as matimg
from tensorflow.keras.datasets import mnist
import pandas as pd
import numpy as np
from PIL import Image
import imageio
import cv2
import os
from sklearn.metrics import accuracy_score

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)

x_train, x_test = x_train / 255-0.5, x_test / 255-0.5

y_train=tf.keras.utils.to_categorical(y_train,10)

y_test=tf.keras.utils.to_categorical(y_test,10)

x_train=np.expand_dims(x_train,axis=-1)

x_test=np.expand_dims(x_test,axis=-1)

model = tf.keras.models.load_model("model_save/model_0.h5")
print('acc:',accuracy_score(np.argmax(y_test,axis=1),np.argmax(model.predict(x_test),axis=1)))