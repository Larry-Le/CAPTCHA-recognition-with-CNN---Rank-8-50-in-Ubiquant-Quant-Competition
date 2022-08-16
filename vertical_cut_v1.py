import cv2
import os
from PIL import Image
import pycapt
from skimage import io,filters,measure,morphology
import numpy as np


# 图片预处理，得到干净的不含噪声和干扰线的代码
def dilate_demo(image):
    # print(image.shape)

    # kernel_1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    image = cv2.bitwise_not(image)
    # image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_1)
    # cv2.imshow("binary", image)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # ret, binary = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    # cv2.imshow("binary", binary)
    # cv2.waitKey(0)

    dst = cv2.dilate(image, kernel)
    dst = cv2.bitwise_not(dst)
    # cv2.imshow("dilate", dst)
    # cv2.waitKey(0)
    return dst


def picture_deal(imagepath):
    if not os.path.isdir(imagepath.split('.')[0]):
        os.mkdir(imagepath.split('.')[0])
    # 以灰度模式读取图片
    gray = cv2.imread(imagepath, 0)
    # cv2.imshow('orgin_image', gray)
    # cv2.waitKey(0)
    # 将图片的边缘变为白色
    height, width = gray.shape
    for i in range(width):
        gray[0, i] = 255
        gray[height-1, i] = 255
    for j in range(height):
        gray[j, 0] = 255
        gray[j, width-1] = 255

    # 中值滤波
    thresh = filters.threshold_otsu(gray)  # 自动确定二值化的阈值
    ret, thresh1 = cv2.threshold(gray, 85, 255, cv2.THRESH_BINARY)

    # cv2.imshow('thresh_image', thresh1)
    # cv2.waitKey(0)
    thresh1 = dilate_demo(thresh1)

    img = Image.fromarray(cv2.cvtColor(thresh1, cv2.COLOR_BGR2RGB))
    img = pycapt.dele_noise(img, N=2, Z=1)

    # img = pycapt.dele_line(img, 8)
    # img = pycapt.dele_line(img, 7)
    # img = pycapt.dele_line(img, 6)
    img = pycapt.dele_line(img, 5)
    img = pycapt.dele_line(img, 4)
    img = pycapt.dele_line(img, 3)
    img = pycapt.dele_line(img, 2)
    img = pycapt.dele_line(img, 1)
    gray = cv2.cvtColor(np.asarray(img),cv2.COLOR_BGR2GRAY)

    # cv2.imshow('delete_image', gray)
    # cv2.waitKey(0)
    # gray = dilate_demo(gray)

    cv2.imwrite(imagepath.split('/')[0] + '_dealt/' + imagepath.split('/')[1].split('.')[0] + '.jpg', gray)

    # contours, hierarch = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # for i in range(len(contours)):
    #     area = cv2.contourArea(contours[i])
    #     print(area)
    #
    #     if area < 8:
    #         cv2.drawContours(gray, [contours[i]],0,0,-1)

    # denoise = cv2.fastNlMeansDenoising(thresh1, templateWindowSize=7,searchWindowSize=21,h=10)

    # img = Image.fromarray(cv2.cvtColor(denoise, cv2.COLOR_BGR2RGB))
    # img = pycapt.dele_noise(img, N=2, Z=1)
    # img = pycapt.dele_line(img, 8)
    # img = pycapt.dele_line(img, 7)
    # img = pycapt.dele_line(img, 6)
    # img = pycapt.dele_line(img, 5)
    # img = pycapt.dele_line(img, 4)
    # img = pycapt.dele_line(img, 3)
    # img = pycapt.dele_line(img, 2)
    # img = pycapt.dele_line(img, 1)


    # gray = cv2.cvtColor(np.asarray(img),cv2.COLOR_BGR2GRAY)
    # cv2.imshow('final_image', gray)
    # cv2.waitKey(0)
    # cv2.imwrite(imagepath.split('/')[0]+'_dealt/'+imagepath.split('/')[1].split('.')[0]+'.jpg',gray)


def picture_deal2(imagepath):
    if len(os.listdir(imagepath.split('/')[0]+'_dealt/'+imagepath.split('/')[1].split('.')[0]))!=0:
        # os.mkdir(imagepath.split('.')[0])
        return
    else:
        gray = cv2.imread(imagepath, 0)
        # cv2.imshow('orgin_image', gray)
        # cv2.waitKey(0)
        # 将图片的边缘变为白色
        # height, width = gray.shape
        # for i in range(width):
        #     gray[0, i] = 255
        #     gray[height - 1, i] = 255
        # for j in range(height):
        #     gray[j, 0] = 255
        #     gray[j, width - 1] = 255

        # 中值滤波
        thresh = filters.threshold_otsu(gray)-50  # 自动确定二值化的阈值
        # color_mean = np.mean(gray)
        ret, thresh1 = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        print(imagepath)
        # cv2.imshow('thresh_image', thresh1)
        # cv2.waitKey(0)

        contours, hierarch = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            # print(area)

            if area < 15:
                cv2.drawContours(thresh1, [contours[i]],0,255,-1)
        # cv2.imshow('thresh_image', thresh1)
        # cv2.waitKey(0)

        thresh1 = dilate_demo(thresh1)

        img = Image.fromarray(cv2.cvtColor(thresh1, cv2.COLOR_BGR2RGB))
        img = pycapt.dele_noise(img, N=2, Z=1)

        # img = pycapt.dele_line(img, 8)
        # img = pycapt.dele_line(img, 7)
        # img = pycapt.dele_line(img, 6)
        img = pycapt.dele_line(img, 5)
        img = pycapt.dele_line(img, 4)
        img = pycapt.dele_line(img, 3)
        img = pycapt.dele_line(img, 2)
        img = pycapt.dele_line(img, 1)
        gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2GRAY)

        # cv2.imshow('delete_image', gray)
        # cv2.waitKey(0)
        # gray = dilate_demo(gray)

        cv2.imwrite(imagepath.split('/')[0] + '_dealt2/' + imagepath.split('/')[1].split('.')[0] + '.jpg', gray)


def vertical(img):
    """传入二值化后的图片进行垂直投影"""
    pixdata = img.load()
    w,h = img.size
    ver_list,standard_list = [],[]
    # 开始投影
    for x in range(w):
        black = 0
        for y in range(h):
            if pixdata[x,y] == 0:
                black += 1
        ver_list.append(black)
    for y in range(h):
        black = 0
        for x in range(w):
            if pixdata[x, y] == 0:
                black += 1
        standard_list.append(black)

    # 判断上下边界
    up = 0
    down = 0
    up_flag,down_flag = False,False
    for i,count in enumerate(standard_list):
        if up_flag is False and count > 15 and i>15:
            up = i - 1
            up_flag = True
        if down_flag is False and up_flag is True and count < 8 and i > 80:
            down = i
            down_flag = True

    if up_flag== False or down_flag== False or down<=up:
        return []

    l,r = 0,0
    flag = False
    cuts = []


    for i,count in enumerate(ver_list):
        # 阈值这里为0
        if flag is False and count > 5:
            l = i
            flag = True
        if flag and count == 0:
            r = i-1
            flag = False
            if (r-l)<3:
                continue
            else:
                cuts.append((l,up,r,down))

    return cuts


def standard(img):
    pixdata = img.load()
    w, h = img.size
    ver_list = []
    # 开始投影
    for y in range(h):
        black = 0
        for x in range(w):
            if pixdata[x, y] == 0:
                black += 1
        ver_list.append(black)
    # 判断边界
    u, d = 0, 0
    flag = False
    cuts = []
    for i, count in enumerate(ver_list):
        # 阈值这里为0
        if flag is False and count > 10:
            u = i
            flag = True
        if flag and count == 0:
            d = i - 1
            flag = False
            cuts.append((u, d))
    return cuts


def remove_dir(dir):
    dir = dir.replace('\\', '/')
    if(os.path.isdir(dir)):
        for p in os.listdir(dir):
            remove_dir(os.path.join(dir,p))
        if(os.path.exists(dir)):
            os.rmdir(dir)
    else:
        if(os.path.exists(dir)):
            os.remove(dir)


def run_vertical_cut(img_file):
    data_type = "Type_1"
    if not os.path.isdir('{}_dealt'.format(data_type)):
        os.mkdir('{}_dealt'.format(data_type))

    # if not os.path.isdir('{}_dealt2'.format(data_type)):
    #     os.mkdir('{}_dealt2'.format(data_type))

    # for i in [i for i in os.listdir('{}'.format(data_type)) if i.find('jpg')>0 ]:
    #     picture_deal('{}/'.format(data_type)+i)

    picture_deal(img_file)

    img_list = [i for i in os.listdir('{}_dealt'.format(data_type)) if i.find('jpg')>0 ]

    for j in img_list:
        # print(j)
        p = Image.open('{}_dealt/'.format(data_type)+j)
        cuts = vertical(p)
        # ii = 0
        # print(j)
        # print(cuts)
        if not os.path.isdir(('{}_dealt/'.format(data_type)+j).split('.')[0]):
            os.mkdir(('{}_dealt/'.format(data_type)+j).split('.')[0])
        else:
            remove_dir(('{}_dealt/'.format(data_type)+j).split('.')[0])
            # os.removedirs(('{}_dealt/'.format(data_type)+j).split('.')[0])
            os.mkdir(('{}_dealt/'.format(data_type) + j).split('.')[0])

        ii=0
        for i, n in enumerate(cuts, 1):
            temp = p.crop((n[0], n[1], n[2], n[3]))  # 调用crop函数进行切割
            # temp = standard(temp)
            # print(np.sum(np.array(temp) < 10) / np.sum(np.array(temp) > 250))
            if (np.sum(np.array(temp) < 10) / np.sum(np.array(temp) > 250)) > 0.2:
                temp.save('{}_dealt/'.format(data_type) + j.split('.')[0] + "/cut%s.png" % ii)
                ii = ii+1
            else:
                pass

    print("run vertical_cut successful!")

    # for i in [i for i in os.listdir('{}'.format(data_type)) if i.find('jpg') > 0]:
    #     picture_deal2('{}/'.format(data_type) + i)

'''
    img_list = [i for i in os.listdir('{}_dealt2'.format(data_type)) if i.find('jpg') > 0]

    for j in img_list:
        p = Image.open('{}_dealt2/'.format(data_type) + j)
        cuts = vertical(p)
        # ii = 0
        # print(j)
        # print(cuts)
        if not os.path.isdir(('{}_dealt/'.format(data_type) + j).split('.')[0]):
            os.mkdir(('{}_dealt/'.format(data_type) + j).split('.')[0])
        else:
            os.removedirs(('{}_dealt/'.format(data_type)+j).split('.')[0])
            os.mkdir(('{}_dealt/'.format(data_type) + j).split('.')[0])

        ii = 0
        for i, n in enumerate(cuts, 1):
            temp = p.crop((n[0], n[1], n[2], n[3]))  # 调用crop函数进行切割
            # temp = standard(temp)
            print(np.sum(np.array(temp)<10)/np.sum(np.array(temp)>250))
            if (np.sum(np.array(temp)<10)/np.sum(np.array(temp)>250)) >0.20:
                temp.save('{}_dealt/'.format(data_type) + j.split('.')[0] + "/cut%s.png" % ii)
                ii = ii+1
            else:
                pass
'''

