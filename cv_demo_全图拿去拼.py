'''
Lucas-Kanade tracker
====================
Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.
Usage
-----
lk_track.py [<video_source>]
Keys
----
ESC - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2

from common import anorm2, draw_str

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=500,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)


class App:
    def __init__(self, video_src):
        self.track_len = 10
        self.detect_interval = 5    # 每5帧检测新的关键点，不延用上一帧的关键点做跟踪
        self.tracks = []
        self.cam = cv2.VideoCapture(video_src)
        self.start_and_now_frame = 0
        self.end_frame = 1500
        self.cam.set(cv2.CAP_PROP_POS_FRAMES, self.start_and_now_frame)
        # self.frame_idx = self.start_frame
        self.first_frame = self.cam.read()[1]
        self.add_result = self.first_frame

    def run(self):
        while self.start_and_now_frame < self.end_frame:
            _ret, frame = self.cam.read()
            if not _ret or self.start_and_now_frame >= self.end_frame:
                break
            # self.start_and_now_frame += 1
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()

            if len(self.tracks) > 0:  # and self.frame_idx % self.detect_interval == 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                print("self.start_and_now_frame:", self.start_and_now_frame)
                # 前一帧的角点和当前帧的图像作为输入来得到角点在当前帧的位置
                p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                # 当前帧跟踪到的角点及图像和前一帧的图像作为输入来找到前一帧的角点位置
                p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)

                d = abs(p0 - p0r).reshape(-1, 2).max(-1)    # 得到角点回溯与前一帧实际角点的位置变化关系
                good = d < 1    # 判断d内的值是否小于1，大于1被认为是错误的跟踪点
                new_tracks = []
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    # 这里是关键点的位置，最多10个，用于可视化光流法中每个关键点后面的运动轨迹
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                self.tracks = new_tracks

                # 开始拼接：只用y变量；因为这个光流法的特征点检测和匹配还是很准的，直接采用前一帧和后一帧的y坐标插值作为移动的距离，直接拼
                diff_y_list = []
                if len(self.tracks) > 1:
                    for one_keyp in self.tracks:
                        one_diff_y = one_keyp[-2][1] - one_keyp[-1][1]
                        diff_y_list.append(one_diff_y)
                    # 优化1：对前2%的移动偏移量做个平均
                    sort_diff_y = sorted(diff_y_list, reverse=True)
                    try:    # 会出现前2%为0的情况，也就是特征点的数量很少，这里就直接拿前面5个的距离
                        cut_scale = int(np.rint(len(sort_diff_y) * 0.02))
                        if cut_scale >= 8:
                            sort_diff_y = sort_diff_y[:cut_scale]
                        else:
                            sort_diff_y = sort_diff_y[:]
                    except:
                        sort_diff_y = sort_diff_y[:5]
                    new_diff_y = int(sum(sort_diff_y)/len(sort_diff_y))
                    diff_y = int(max(diff_y_list))
                    diff_y = new_diff_y
                    print("diff_y:", diff_y)
                    if diff_y > 0:
                        crop_next_roi = frame[frame.shape[0]-diff_y:, :]
                        new_mask = np.zeros((self.add_result.shape[0]+diff_y, self.add_result.shape[1], 3), np.uint8)
                        new_mask[0:self.add_result.shape[0], :] = self.add_result
                        new_mask[self.add_result.shape[0]:, :] = crop_next_roi
                        self.add_result = new_mask.copy()

                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))

            if self.start_and_now_frame % self.detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)    # shi-Tomasi角点检测
                # print("p:", p)
                # print("p reshape:", np.float32(p).reshape(-1, 2))
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])
                # print("self.tracks:", self.tracks)

            self.start_and_now_frame += 1
            self.prev_gray = frame_gray
            cv2.imshow('lk_track', vis)
            cv2.imshow('self.add_result', self.add_result)

            ch = cv2.waitKey(4)
            if ch == 27:
                break
        cv2.imwrite("./kjg_01.png", self.add_result)


def main():
    import sys
    try:
        # video_src = sys.argv[1]
        index = '1'
        # video_src = rf"F:\1_sheng\image_stitch\机柜\机柜{index}\机柜{index}视频_没有后面的部分.mp4"
        # video_src = rf"F:\1_sheng\image_stitch\机柜\机柜{index}\机柜{index}视频.mp4"
        video_src = rf"F:\1_sheng\image_stitch\机柜-2021-05-22\关门关灯\可见光.mp4"
        # video_src = rf"F:\1_sheng\image_stitch\42U机房正门\4.关门,开灯\3.开门，开灯.mp4"
        # video_src = rf"F:\1_sheng\image_stitch\0513_video_to_frame\WeChat_20210513110458.mp4"
    except:
        video_src = 0

    App(video_src).run()
    print('Done')


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
    # a = 3
    # print("1", id(a))
    # def Fuc():
    #     global a
    #     print("2", id(a))
    #     a = a + 1
    # Fuc()
    # print("3", id(a))
