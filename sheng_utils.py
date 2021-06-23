import time
import cv2
import numpy
import imageio
import skimage
import numpy as np
import matplotlib.pyplot as plt


def cut_video():
    video_path = r"F:\1_sheng\image_stitch\42U机房正门\1.关门,关灯\1关门-关灯.mp4"
    cap = cv2.VideoCapture(video_path)
    start_frame = 200
    cut_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    save_video_path = r"F:\1_sheng\image_stitch\42U机房正门\1.关门,关灯\调整后的视频.mp4"
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    print((cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(save_video_path, fourcc, 30.0,
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    # exit(0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    while start_frame < cut_frame:
        _, frame = cap.read()
        if _:

            out.write(frame)

            cv2.imshow("frame", frame)
            cv2.waitKey(5)
            print(start_frame)
            start_frame += 1
        else:
            break
    out.release()
    cap.release()


def imageio_cut_video():
    folder = r'4.关门,开灯'
    # name = folder.replace(",", "-").replace()
    video_path = rf"F:\1_sheng\image_stitch\42U机房正门\{folder}\4.关门，开灯.mp4"
    vid = imageio.get_reader(video_path, 'ffmpeg')

    save_video_path = rf"F:\1_sheng\image_stitch\42U机房正门\{folder}\调整后的视频.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(save_video_path, fourcc, 30.0, (1920, 1080))

    start_frame = 75
    end_frame = 1900
    now_frame_idx = 0
    for num, im in enumerate(vid):
        # image的类型是mageio.core.util.Image可用下面这一注释行转换为arrary
        image = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
        if (start_frame < now_frame_idx) and (now_frame_idx <= end_frame):
            out.write(image)
            cv2.imshow("frame", image)
            cv2.waitKey(5)
            print("now_frame_idx:", now_frame_idx)
        else:
            print("no coming:", now_frame_idx)
        now_frame_idx += 1
    out.release()


def test_fft():
    data = [1, 4, 6, 8, 9, 8, 6, 4, 1]
    print(len(data))
    data = np.asarray(data, dtype=np.float)
    wave = np.cos(data)
    transformed = np.fft.fft(wave)  # 傅里叶变换
    plt.plot(transformed)  # 绘制变换后的信号
    plt.show()


if __name__ == '__main__':
    # cut_video()
    # imageio_cut_video()
    # aa = [[1, 2], [3, 4]]
    # bb = [[1, 2], [3, 6]]
    test_fft()
    print("~_~")
