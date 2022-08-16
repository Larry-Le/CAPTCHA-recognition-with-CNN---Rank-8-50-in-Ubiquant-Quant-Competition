import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as matimg
import pandas as pd
import numpy as np
from PIL import Image
import imageio
import cv2
import os


# 返回目录下的所有目录
def load_dir(current_dir):
    res = []
    file_list = os.listdir(current_dir)
    for f in file_list:
        current_path = current_dir + '/' + f
        if os.path.isdir(current_path):
            res.append(current_path)
        else:
            pass
    return res


# 返回目录下所有以"cut"开头的图片
def load_pictures_from_dir(dir_list):
    res = {}
    for dir in dir_list:
        picture_number = 0
        file_list = os.listdir(dir)
        for file in file_list:
            if file[: 3] == 'cut':
                picture_number += 1
            else:
                pass
#        picture_number = len(os.listdir(dir))
        res[dir] = ["cut"+str(i)+".png" for i in range(picture_number)]
    return res


# model.predict()返回的是一个数组，这里我们得到最终的结果（最大值所对应的索引）
def get_model_output(output_list):
    max_number = max(output_list[0])
    for i, num in enumerate(output_list[0]):
        if num == max_number:
            return i
        else:
            pass
    return


def get_model_output_type4(output_list):
    model_dict = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
           10: "A", 11: "B", 12: "C", 13: "D", 14: "E", 15: "F", 16: "G", 17: "H", 18: "I",
           19: "J", 20: "K", 21: "L", 22: "M", 23: "N", 24: "O", 25: "P", 26: "Q", 27: "R",
           28: "S", 29: "T", 30: "U", 31: "V", 32: "W", 33: "X", 34: "Y", 35: "Z", 36: "a",
           37: "b", 38: "c", 39: "d", 40: "e", 41: "f", 42: "g", 43: "h", 44: "i", 45: "j",
           46: "k", 47: "l", 48: "m", 49: "n", 50: "o", 51: "p", 52: "q", 53: "r", 54: "s",
           55: "t", 56: "u", 57: "v", 58: "w", 59: "x", 60: "y", 61: "z"}
    max_number = max(output_list[0])
    for i, num in enumerate(output_list[0]):
        if num == max_number:
            return model_dict[i]
        else:
            pass
    return


# 这里我们对图片进行了处理，使其变为像素是28*28的图片
def change_pictures(pictures_dic):
    for dir_path in pictures_dic:
        imgs = pictures_dic[dir_path]
        for img_path in imgs:
            img = matimg.imread(dir_path + '/' + img_path)
            img = np.array(img)
            print(img.shape)
            img_new = np.array([1] * (160 ** 2)).reshape(160, 160)
            if img.shape[1] % 2 == 0:
                img_new[:, 80-int(img.shape[1]/2): 80+int(img.shape[1]/2)] = img
            elif img.shape[1] % 2 == 1 and img.shape[1] > 1:
                img_new[:, 80-int((img.shape[1]+1)/2): 79+int((img.shape[1]+1)/2)] = img
            else:
                img_new[:, 80:81] = img
            imageio.imwrite(dir_path + '/new_' + img_path, img_new)

#           img2 = Image.open(dir_path + '/new_' + img_path)
            img2 = cv2.imread(dir_path + '/new_' + img_path)
            out = cv2.resize(img2, (28, 28))
            print(type(out))
            imageio.imwrite(dir_path + '/new_' + img_path, out)
    print("successfully changed pictures")
    return


# 对于高度不为160的图片的处理方法
def change_pictures_1(pictures_dic):
    for dir_path in pictures_dic:
        imgs = pictures_dic[dir_path]
        for img_path in imgs:
            img = matimg.imread(dir_path + '/' + img_path)

                # img = np.array(img)
                # img = cv2.imread(dir_path+'/'+img_path,0)

            height, width = img.shape

            # print(img.shape, dir_path, img_path)
            if height == width:
                pass
            elif height > width:
                img_new = np.array([1] * (height ** 2)).reshape(height, height)
                if width % 2 == 0:
                    img_new[:, int(height / 2) - int(width / 2): int(height / 2) + int(width / 2)] = img
                elif width % 2 == 1 and width > 1:
                    img_new[:, int(height / 2) - int((width + 1) / 2): int(height / 2) - 1 + int((width + 1) / 2)] = img
                else:
                    img_new[:, int(height / 2): int(height / 2) + 1] = img

            else:
                img_new = np.array([1] * (width ** 2)).reshape(width, width)
                if height % 2 == 0:
                    img_new[int(width / 2) - int(height / 2): int(width / 2) + int(height / 2), :] = img
                elif height % 2 == 1 and height > 1:
                    img_new[int(width / 2) - int((height + 1) / 2): int(width / 2) - 1 + int((height + 1) / 2), :] = img
                else:
                    img_new[int(width / 2): int(width / 2) + 1, :] = img
            imageio.imwrite(dir_path + '/new_' + img_path, img_new)
#           img2 = Image.open(dir_path + '/new_' + img_path)
            img2 = cv2.imread(dir_path + '/new_' + img_path,0)
            out = cv2.resize(img2, (28, 28))
            # print(type(out))
            out = cv2.bitwise_not(out)
            imageio.imwrite(dir_path + '/new_' + img_path, out)
    print("successfully changed pictures")
    return


# 对于一个目录中的每一个图片进行预测，然后把结果拼接在一起转换成csv文件
def run_model(pictures_dic, model):
    for dir_path in pictures_dic:
        imgs = pictures_dic[dir_path]
        model_out = []
        for img_path in imgs:
            # print(dir_path, img_path)
            img = cv2.imread(dir_path + "/new_" + img_path)
            img = np.array([img[:, :, 0]])
            img = img / 255 - 0.5
            img = np.expand_dims(img, axis=-1)
            res = model.predict(img)
            out_res = get_model_output_type4(res)
            model_out.append(out_res)

        model_out_df = pd.DataFrame([''.join([str(x) for x in model_out])])
        model_out_df.to_csv(dir_path + '/' + 'model_out.csv')
    # print("run model successful")
    pass


def load_data():
    # 加载保存的.h5模型
    data_path = os.getcwd()
    model = tf.keras.models.load_model("guangchi_submit/UBI_ocr_net/model_save/model_2.h5")
    dir_list = load_dir(data_path+"/Type_1_dealt")
    pictures_dic = load_pictures_from_dir(dir_list)
    change_pictures_1(pictures_dic)
    run_model(pictures_dic, model)
    print("load data and run model successful!")
