import cv2
import os
import numpy as np
import glob
from PIL import Image
import random
import time
import sys
import matplotlib.pyplot as plt


def cv_imread(file_path):
    # 读取中文路径下的图片
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    if cv_img.shape[-1] == 4:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGBA2RGB)
    return cv_img


def detect(image):
    # 转化为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 创建SIFT生成器
    # descriptor是一个对象，这里使用的是SIFT算法
    descriptor = cv2.xfeatures2d.SIFT_create()
    # 检测特征点及其描述子（128维向量）
    kps, features = descriptor.detectAndCompute(image, None)
    return kps, features


def show_points(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    descriptor = cv2.SIFT_create()
    image = cv2.GaussianBlur(image, (5, 5), 0)
    kps, features = descriptor.detectAndCompute(image, None)
    print(f"特征点数：{len(kps)}")
    img_left_points = cv2.drawKeypoints(image, kps, image)
    plt.figure(figsize=(9,9))
    plt.imshow(img_left_points)
    plt.show()


def rotate_bound(image, angle):
    # 获取图像的尺寸
    # 旋转中心
    (h, w) = image.shape[:2]
    (cx, cy) = (w / 2, h / 2)

    # 设置旋转矩阵
    M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # 计算图像旋转后的新边界
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # 调整旋转矩阵的移动距离（t_{x}, t_{y}）
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    return cv2.warpAffine(image, M, (nW, nH))


def main_2():
    img_1 = cv_imread(r"./抓图图示/0位置/5位置.jpg")
    img_2 = cv_imread(r"./抓图图示/25位置/20位置.jpg")
    # kps, features = detect(img_1)
    img_flip = rotate_bound(img_1, 270)
    show_points(img_1)
    # cv2.namedWindow("test_stitch", 0)
    # cv2.resizeWindow("test_stitch", 1000, 1000)
    cv2.imshow("img_1", img_1)
    cv2.imshow("img_flip", img_flip)
    cv2.waitKey(0)


def main_1():
    t1 = time.time()
    # img_1 = cv_imread(r"./抓图图示/0位置/0位置(最高）.jpg")
    img_1 = cv_imread(r"./抓图图示/0位置/5位置.jpg")
    img_2 = cv_imread(r"./抓图图示/25位置/20位置.jpg")
    img_3 = cv_imread(r"./抓图图示/50位置/45位置.jpg")
    img_4 = cv_imread(r"./抓图图示/75位置/70位置.jpg")
    img_5 = cv_imread(r"./抓图图示/100位置/100位置（最低）.jpg")
    output = 'result3.jpg'

    # img_1 = rotate_bound(img_1, 270)
    # img_2 = rotate_bound(img_2, 270)
    # img_3 = rotate_bound(img_3, 270)
    # img_1 = cv2.GaussianBlur(img_1, (5, 5), 0)
    # img_2 = cv2.GaussianBlur(img_2, (5, 5), 0)
    # cv2.imshow("img_1", img_1)
    # cv2.imshow("img_2", img_2)
    # cv2.imshow("img_3", img_3)
    # cv2.waitKey(0)

    imgs = []
    imgs.append(img_1)
    imgs.append(img_2)
    imgs.append(img_3)
    imgs.append(img_4)
    imgs.append(img_5)

    stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)  # cv.Stitcher_SCANS , STITCHER_PANORAMA
    status, pano = stitcher.stitch(imgs)
    if status != cv2.Stitcher_OK:
        print("Can't stitch images, error code = %d" % status)
        sys.exit(-1)
    cv2.imwrite(output, pano)
    print("stitching completed successfully. %s saved!" % output)

    print("拼接耗时：", time.time()-t1)
    cv2.namedWindow("test_stitch", 0)
    cv2.resizeWindow("test_stitch", 1000, 1000)
    cv2.imshow("test_stitch", output)
    cv2.waitKey(0)


if __name__ == '__main__':
    main_1()
