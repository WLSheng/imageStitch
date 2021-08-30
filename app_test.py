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
    url = "http://192.168.1.244:8700/imageStitch"  # 后端api链接
    # f = open(r"F:\1_sheng\image_stitch\test_send.mp4", 'rb')  # 以二进制打开前端本地文件
    # f = open(r"F:\1_sheng\image_stitch\机柜\机柜2\机柜2视频.mp4", 'rb')  # 以二进制打开前端本地文件
    userdata = json.dumps({'videoPath': r"F:\1_sheng\image_stitch\0811\kejianguan.mp4"})  # 将二进制文件封装为这样一个字典，索引为file
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


def clear():
    import datetime
    # dir_list = os.listdir("./cache/")
    dir_list = os.listdir("./cache/")
    if not dir_list:
        print("没有找到缓存的拼接图")
    else:
        # 注意，这里使用lambda表达式，将文件按照最后修改时间顺序升序排列
        # os.path.getmtime() 函数是获取文件最后修改时间
        # os.path.getctime() 函数是获取文件最后创建时间
        for i, one_result_name in enumerate(dir_list):
            print(i, one_result_name)
            img_path = os.path.join(r'./cache/', one_result_name)
            today = datetime.datetime.now()
            # 计算偏移量,前3天
            offset = datetime.timedelta(days=-3)
            # 获取想要的日期的时间,即前3天时间
            re_date = (today + offset)
            # 前3天时间转换为时间戳
            re_date_unix = time.mktime(re_date.timetuple())
            # print("当前日期", today.strftime('%Y-%m-%d'))  # 当前日期
            # print("前3天日期", re_date.strftime('%Y-%m-%d'))  # 前3天日期

            file_time = os.path.getmtime(img_path)  # 文件修改时间
            timeArray = time.localtime(file_time)  # 时间戳->结构化时间
            otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)  # 格式化时间
            print("文件修改时间", otherStyleTime)
            if file_time <= re_date_unix:
                print("已经超过3天,需要删除")
            else:
                print("未超过3天,无需处理!")


if __name__ == '__main__':
    test_send_zip()
    # clear()
