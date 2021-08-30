# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 15:43:08 2018

@author: x

玩一下 cv2.calcOpticalFlowFarneback

https://blog.csdn.net/u014657795/article/details/78764157

"""

import cv2
import numpy as np
import traceback
import os


def find_sobel(gray):
    sobel_type = cv2.CV_64F
    sobel_size = 1  # 一/二阶导数
    sobel_ksize = 7
    rgb_gx = cv2.Sobel(gray, sobel_type, sobel_size, 0, ksize=sobel_ksize, scale=1.0, delta=0.0, borderType=cv2.BORDER_REPLICATE)
    rgb_gy = cv2.Sobel(gray, sobel_type, 0, sobel_size, ksize=sobel_ksize, scale=1.0, delta=0.0, borderType=cv2.BORDER_REPLICATE)

    _rgb_gx = np.expand_dims(rgb_gx, axis=-1)
    _rgb_gy = np.expand_dims(rgb_gy, axis=-1)
    rgb_gxy = abs(np.maximum(rgb_gx, rgb_gy))
    rgb_gxy = rgb_gxy / np.max(rgb_gxy) * 255.0
    rgb_gxy = np.asarray(rgb_gxy, dtype=np.uint8)
    # plt.imshow(rgb_gxy)
    # plt.show()
    return rgb_gxy


class BatchImg:
    def __init__(self, batch_path):
        self.batch_path = batch_path
        self.img_name_list = sorted(os.listdir(batch_path))
        print(self.img_name_list)
        self.frame_now_num = 0
        self.all_frame = len(self.img_name_list)

    def read(self):
        try:
            _img_path = os.path.join(self.batch_path, self.img_name_list[self.frame_now_num])
            img = cv2.imread(_img_path, 0)
            img = find_sobel(img)
            self.frame_now_num += 1
            print(_img_path)
            return True, img
        except:
            print(f"批量图片读取遇到bug:{traceback.format_exc()}")
            return False, None


def draw_flow(im, flow, step=16):
    h, w = im.shape[:2]
    #    global y, x
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1)
    y = y.astype(np.uint16)
    x = x.astype(np.uint16)

    fx, fy = 2 * flow[y, x].T

    # create line endpoints
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines)

    # create image and draw
    #    vis = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    #    vis = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    vis = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)

    for (x1, y1), (x2, y2) in lines:
        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


# input_path = "D:\\xiamao\\xmProject\\xmimg\\dev\\video\\xm2\\video1\\3.mp4"
# input_path = 0
# cap = cv2.VideoCapture(input_path)

video_src = r"F:\1_sheng\image_stitch\img2jpg\30"
cap = BatchImg(video_src)
scale = 1.0

ret, prev_gray = cap.read()
# im = cv2.resize(im, (0, 0), fx=scale, fy=scale)

# prev_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

while True:
    # get grayscale image
    ret, gray = cap.read()
    # im = cv2.resize(im, (0, 0), fx=scale, fy=scale)

    if ret is False:
        break

    # gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # compute flow
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, pyr_scale=0.5, levels=1, winsize=500, iterations=3, poly_n=7, poly_sigma=1.2, flags=1)
    # prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags
    prev_gray = gray

    # plot the flow vectors
    cv2.imshow('Optical flow', draw_flow(gray, flow))  # gray, flow
    if cv2.waitKey(1000) == 27:
        break
