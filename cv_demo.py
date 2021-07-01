from __future__ import print_function

import numpy as np
import cv2
import time
import os
import traceback


lk_params = dict(winSize=(15, 15),  # 从下一帧中在金字塔找指定窗口大小的特征，增大些有益于匹配关键点，相机移动过快需要调大些
                 maxLevel=2,        # 金字塔数，0为不使用，
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))  # 迭代搜索算法的终止准则，最大迭代数和迭代半径？

feature_params = dict(maxCorners=500,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)


class App:
    def __init__(self, video_src):
        print(f"拼接服务收到路径：{video_src}")
        self.video_src = video_src
        self.track_len = 5
        self.detect_interval = 5    # 每5帧检测新的关键点，不延用上一帧的关键点做跟踪
        self.tracks = []
        self.cam = cv2.VideoCapture(video_src)
        self.start_and_now_frame = 0
        self.end_frame = self.cam.get(cv2.CAP_PROP_FRAME_COUNT)
        self.cam.set(cv2.CAP_PROP_POS_FRAMES, self.start_and_now_frame)
        self.all_frmae_num = self.cam.get(cv2.CAP_PROP_FRAME_COUNT)
        # self.frame_idx = self.start_frame
        self.first_frame = self.cam.read()[1]
        print("self.end_frame:", self.end_frame, self.first_frame.shape)
        self.crop_para = 250     # 用于优化拼接速度的，从原图1/3位置往前的200行像素开始，到1/3位置往往后200行像素结束,这个参数可调
        if int(self.first_frame.shape[0]/3 - self.crop_para) < 0:
            self.crop_para = int(self.first_frame.shape[0]/3)
            print("裁剪参数会超出视频帧的范围，把裁剪参数设为帧高的1/3")
        self.col_para = 2        # 3是从中间为开始，x坐标往前后抠1/3像素做匹配，最低为2
        self.first_up_half_img = self.first_frame[0: int(self.first_frame.shape[0]/3 - self.crop_para), :].copy()      # 用于最后拼在最上面
        self.width = self.first_frame.shape[1]
        self.add_result = self.first_frame[int(self.first_frame.shape[0]/3 - self.crop_para): int(self.first_frame.shape[0]/3 + self.crop_para), :].copy()
        self.first_frame = self.first_frame[int(self.first_frame.shape[0]/3 - self.crop_para): int(self.first_frame.shape[0]/3 + self.crop_para),
                           int(self.width/2 - int(self.width/self.col_para)):int(self.width/2 + int(self.width/self.col_para))].copy()
        self.last_frame = None
        self.jump_frame = 1
        self.prev_frame = None

    def run(self, debug=False):
        print("开始拼接......")
        # while self.start_and_now_frame < self.end_frame:
        while self.start_and_now_frame < 6000:
            _ret, org_frame = self.cam.read()
            if _ret:
                self.prev_frame = org_frame.copy()
            if self.start_and_now_frame % self.jump_frame == 0:
                t1 = time.time()
                if not _ret or self.start_and_now_frame >= 6000:
                    print("_ret:", _ret, " , self.start_and_now_frame:", self.start_and_now_frame)
                    print("视频流有问题，读不到图片了，这里拿前一帧作为最后一帧，拼到最后面去")
                    self.last_frame = self.prev_frame
                    print(" 开始拼最后一张 ")
                    crop_next_roi = self.last_frame[int(self.last_frame.shape[0] / 3)+self.crop_para:, :]
                    new_mask = np.zeros((self.add_result.shape[0] + crop_next_roi.shape[0], self.add_result.shape[1], 3), np.uint8)
                    new_mask[0:self.add_result.shape[0], :] = self.add_result
                    new_mask[self.add_result.shape[0]:, :] = crop_next_roi
                    self.add_result = new_mask.copy()
                    break

                if self.start_and_now_frame == int(self.all_frmae_num - self.detect_interval - 2):
                    self.last_frame = org_frame
                    print("---- coming last frame -----")
                temp_crop_frame = org_frame[int(org_frame.shape[0]/3 - self.crop_para):int(org_frame.shape[0]/3 + self.crop_para), :].copy()
                frame = org_frame[int(org_frame.shape[0]/3 - self.crop_para):int(org_frame.shape[0]/3 + self.crop_para),
                            int(self.width/2 - int(self.width/self.col_para)):int(self.width/2 + int(self.width/self.col_para))].copy()
                # self.start_and_now_frame += 1
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                vis = frame.copy()

                if len(self.tracks) > 0:  # and self.frame_idx % self.detect_interval == 0:
                    img0, img1 = self.prev_gray, frame_gray
                    p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                    # 前一帧的角点和当前帧的图像作为输入来得到角点在当前帧的位置
                    p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                    # 当前帧跟踪到的角点及图像和前一帧的图像作为输入来找到前一帧的角点位置
                    p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)

                    # print("找角点一张耗时：", time.time() - t1)
                    d = abs(p0 - p0r).reshape(-1, 2).max(-1)    # 得到角点回溯与前一帧实际角点的位置变化关系
                    good = d < 1    # 判断d内的值是否小于1，大于1被认为是错误的跟踪点
                    new_tracks = []
                    # tc = time.time()
                    for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                        if not good_flag:
                            continue
                        tr.append((x, y))
                        # 这里是关键点的位置，最多10个，用于可视化光流法中每个关键点后面的运动轨迹
                        if len(tr) > self.track_len:
                            del tr[0]
                        new_tracks.append(tr)
                        # cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                    # print("画点耗时：", time.time() - tc)
                    self.tracks = new_tracks

                    # 开始拼接：只用y变量；因为这个光流法的特征点检测和匹配还是很准的，直接采用前一帧和后一帧的y坐标插值作为移动的距离，直接拼
                    diff_y_list = []
                    # print("len(self.tracks):", len(self.tracks))
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
                        # if self.start_and_now_frame == 446:
                        #     print(sort_diff_y, len(sort_diff_y))
                        new_diff_y = int(sum(sort_diff_y)/len(sort_diff_y))     # 求前2%的y方向平均偏移
                        # diff_y = int(max(diff_y_list))    # 直接拿y方向的最大偏移
                        diff_y = new_diff_y
                        print(f"self.all_frmae_num:{self.all_frmae_num}, self.start_and_now_frame:{self.start_and_now_frame}, "
                              f"diff_y:{diff_y}")
                        if diff_y > 0 and self.last_frame is None:
                            crop_next_roi = temp_crop_frame[frame.shape[0]-diff_y:, :]
                            new_mask = np.zeros((self.add_result.shape[0]+diff_y, self.add_result.shape[1], 3), np.uint8)
                            new_mask[0:self.add_result.shape[0], :] = self.add_result
                            new_mask[self.add_result.shape[0]:, :] = crop_next_roi
                            self.add_result = new_mask.copy()
                        # 第一阶段优化：用前半张图片进行拼接，拼到最后一张时再用最后一张直接全部拼上去
                        if self.last_frame is not None:
                            print(" 开始拼最后一张 ")
                            crop_next_roi = self.last_frame[int(self.last_frame.shape[0] / 3)+self.crop_para:, :]
                            new_mask = np.zeros((self.add_result.shape[0]+crop_next_roi.shape[0], self.add_result.shape[1], 3), np.uint8)
                            new_mask[0:self.add_result.shape[0], :] = self.add_result
                            new_mask[self.add_result.shape[0]:, :] = crop_next_roi
                            self.add_result = new_mask.copy()
                            break
                    # print("后处理拼一张上去耗时：", time.time() - t1)

                    cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                    # draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))

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
            if debug:
                cv2.imshow('lk_track', vis)
                cv2.imshow('self.add_result', self.add_result)
                ch = cv2.waitKey(1)
                if ch == 27:
                    break
        # 把前半部分的也加上去
        new_mask = np.zeros((self.add_result.shape[0]+self.first_up_half_img.shape[0], self.add_result.shape[1], 3), np.uint8)
        new_mask[0:self.first_up_half_img.shape[0], :] = self.first_up_half_img
        new_mask[self.first_up_half_img.shape[0]:, :] = self.add_result
        try:
            write_name = os.path.split(self.video_src)[1].split(".")[0] + ".png"
        except:
            print("制作路径发生未知的bug:", traceback.format_exc())
            write_name = str(time.strftime("%Y_%m_%d-%H_%M_%S")) + ".png"

        cv2.imwrite(f"./cache/{write_name}", self.add_result)
        print("保存视频成功，路径：", write_name)
        # cv2.imwrite(f"./optimize_cost_time_5.png", new_mask)
        return new_mask


def main():
    import sys
    try:
        # video_src = sys.argv[1]
        index = '4'
        # video_src = rf"F:\1_sheng\image_stitch\机柜\机柜{index}\机柜{index}视频.mp4"
        # video_src = rf"F:\1_sheng\image_stitch\机柜\机柜2\机柜2视频.mp4"
        video_src = r"F:\1_sheng\image_stitch\0630视频\c21ce7ae-481a-4246-b401-feb07a00fce8_193915.mp4"
        # video_src = rf"F:\1_sheng\image_stitch\42U机房正门\4.关门,开灯\3.开门，开灯.mp4"
        # video_src = rf"F:\1_sheng\image_stitch\0513_video_to_frame\WeChat_20210513110458.mp4"
    except:
        video_src = 0

    App(video_src).run(debug=True)
    print('Done')


if __name__ == '__main__':
    t1 = time.time()
    main()
    print("耗时：", time.time() - t1)
    cv2.destroyAllWindows()
    # a = 3
    # print("1", id(a))
    # def Fuc():
    #     global a
    #     print("2", id(a))
    #     a = a + 1
    # Fuc()
    # print("3", id(a))
