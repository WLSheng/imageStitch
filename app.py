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

app = flask.Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024
# 上面的代码会把尺寸限制为 200 M 。如果上传了大于这个尺寸的文件， Flask 会抛 出一个 RequestEntityTooLarge 异常。


@app.route('/imageStitch', methods=['POST'])
def imageStitch():
    try:
        # para = json.loads(flask.request.get_data())
        # org_video = para['video_file']
        file = flask.request.files.get('video_file')
        save_name = "./cache/" + time.strftime("%Y_%m_%d-%H_%M_%S") + ".mp4"
        file.save(save_name)
        print(f'============= image stitch get mp4 file ======= name: {save_name} ==================')
        time.sleep(1)
        status = '1'    # 1为异常，0为正常
        return_img = cv_demo.App(save_name).run(debug=False)
        retval, buffer = cv2.imencode('.jpg', return_img)
        pic = base64.b64encode(buffer)
        pic = pic.decode()
        returndata = {"status": '0', "Pic": pic}
        print(" =====  拼接结束  ======")
        return json.dumps(returndata)
    except Exception as e:
        print(" 不知道的bug:", traceback.format_exc())
        returndata = {"status": '1', "Pic": ""}
        return json.dumps(returndata)


def start_tornado(port=8700):
    http_server = tornado.httpserver.HTTPServer(tornado.wsgi.WSGIContainer(app))
    http_server.listen(port, address="0.0.0.0")
    print(f"Tornado server starting on port {port}")
    tornado.ioloop.IOLoop.instance().start()


if __name__ == '__main__':
    start_tornado(port=8700)
