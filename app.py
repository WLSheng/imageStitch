import cv_demo
import flask
import tornado.httpserver
import tornado.wsgi
import json
from threading import Thread
import base64
import cv2
import time
import traceback
import os
import datetime

app = flask.Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024
# 上面的代码会把尺寸限制为 200 M 。如果上传了大于这个尺寸的文件， Flask 会抛 出一个 RequestEntityTooLarge 异常。


@app.route('/imageStitch', methods=['POST'])
def imageStitch():
    try:
        para = json.loads(flask.request.get_data())
        videoPath = para['videoPath']
        # file = flask.request.files.get('video_file')
        # save_name = "./cache/" + time.strftime("%Y_%m_%d-%H_%M_%S") + ".mp4"
        # file.save(save_name)
        print(f'============= image stitch get mp4 file ======= path: {videoPath} ==================')
        # time.sleep(1)
        status = '1'    # 1为异常，0为正常
        return_img = cv_demo.App(videoPath).run(debug=False)
        retval, buffer = cv2.imencode('.jpg', return_img)
        pic = base64.b64encode(buffer)
        pic = pic.decode()
        returndata = {"status": '0', "Pic": pic}
        print(" =====  拼接结束  ======")
        try:
            print("开始清理三天前的拼接图")
            dir_list = os.listdir("./cache/")
            if not dir_list:
                print("没有找到缓存的拼接图")
            else:
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

                    file_time = os.path.getmtime(img_path)  # 文件修改时间
                    timeArray = time.localtime(file_time)  # 时间戳->结构化时间
                    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)  # 格式化时间
                    print("文件修改时间", otherStyleTime)
                    if file_time <= re_date_unix:
                        print(f"已经超过3天,需要删除,delete:{img_path}")
                        os.remove(img_path)
                    else:
                        print("未超过3天,无需处理!!!")

        except:
            print(f"清理三天前的拼接图失败，报错：{traceback.format_exc()}")
        return json.dumps(returndata)
    except Exception as e:
        print(" 不知道的bug:", traceback.format_exc())
        returndata = {"status": '1', "Pic": ""}
        return json.dumps(returndata)


def start_tornado(port=8700):
    # http_server = tornado.httpserver.HTTPServer(tornado.wsgi.WSGIContainer(app))
    # http_server.listen(port, address="0.0.0.0")
    # print(f"Tornado server starting on port {port}")
    # tornado.ioloop.IOLoop.instance().start()
    app.run('0.0.0.0', port=port, threaded=True)


if __name__ == '__main__':
    start_tornado(port=8700)
