import cv_demo
import flask
import tornado.httpserver
import tornado.wsgi
import json
from threading import Thread
import base64
import cv2
import numpy as np
import time
app = flask.Flask(__name__)
import os
import zipfile, requests


def test_send_zip():
    url = "http://127.0.0.1:8700/imageStitch"  # 后端api链接
    # f = open(r"F:\1_sheng\image_stitch\test_send.mp4", 'rb')  # 以二进制打开前端本地文件
    f = open(r"F:\1_sheng\image_stitch\机柜\机柜2\机柜2视频.mp4", 'rb')  # 以二进制打开前端本地文件
    userdata = json.dumps({'videoPath': r"F:\1_sheng\image_stitch\test_path.mp4"})  # 将二进制文件封装为这样一个字典，索引为file
    t1 = time.time()
    print("开始发送拼接视频")
    r = requests.post(url=url, data=userdata)
    para = json.loads(r.content)
    print(para['status'])
    # print(para['pic'])
    pic = para['Pic']
    user_image = base64.b64decode(pic)
    img_array = np.fromstring(user_image, np.uint8)
    img = cv2.imdecode(img_array, cv2.COLOR_BGR2RGB)
    print(" ==== 远程拼接耗时：", time.time() - t1)
    cv2.imshow("re", img)
    cv2.waitKey(0)


if __name__ == '__main__':
    test_send_zip()
