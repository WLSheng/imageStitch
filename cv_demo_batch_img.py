from __future__ import print_function

import numpy as np
import cv2
import time
import os
import traceback
from threading import Thread

lk_params = dict(winSize=(30, 30),  # 从下一帧中在金字塔找指定窗口大小的特征，增大些有益于匹配关键点，相机移动过快需要调大些
                 maxLevel=5,  # 金字塔数，0为不使用，
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))  # 迭代搜索算法的终止准则，最大迭代数和迭代半径？

feature_params = dict(maxCorners=200,
                      qualityLevel=0.3,
                      minDistance=15,
                      blockSize=15)


def find_sobel(gray):
    sobel_type = cv2.CV_64F
    sobel_size = 1  # 一/二阶导数
    sobel_ksize = 7
    rgb_gx = cv2.Sobel(gray, sobel_type, sobel_size, 0, ksize=sobel_ksize, scale=1.0, delta=0.0,
                       borderType=cv2.BORDER_REPLICATE)
    rgb_gy = cv2.Sobel(gray, sobel_type, 0, sobel_size, ksize=sobel_ksize, scale=1.0, delta=0.0,
                       borderType=cv2.BORDER_REPLICATE)

    _rgb_gx = np.expand_dims(rgb_gx, axis=-1)
    _rgb_gy = np.expand_dims(rgb_gy, axis=-1)
    rgb_gxy = abs(np.maximum(rgb_gx, rgb_gy))
    rgb_gxy = rgb_gxy / np.max(rgb_gxy) * 255.0
    rgb_gxy = np.asarray(rgb_gxy, dtype=np.uint8)
    rgb_gxy = cv2.cvtColor(rgb_gxy, cv2.COLOR_GRAY2RGB)
    # plt.imshow(rgb_gxy)
    # plt.show()
    return rgb_gxy


class BatchImg:
    def __init__(self, batchimg):
        self.org_img0, self.org_img1 = batchimg
        self.frame_now_num = 0
        self.all_frame_num = self.org_img0.shape[0]/3
        self.crop_start_y = 300
        # self.org_img0 = cv2.imread(r'F:\1_sheng\image_stitch\img2jpg\10_max\1225,400000.jpg')
        fscale = 0.3
        self.org_img0 = cv2.resize(self.org_img0, (0, 0), fx=fscale, fy=fscale)
        self.img0 = self.org_img0.copy()
        # self.org_img1 = cv2.imread(r'F:\1_sheng\image_stitch\img2jpg\10_max\1045,400000.jpg')
        self.org_img1 = cv2.resize(self.org_img1, (0, 0), fx=fscale, fy=fscale)
        self.img1 = self.org_img1.copy()

    def read(self):
        try:
            black_img = np.zeros(self.org_img0.shape, dtype=np.uint8)
            black_img[:self.org_img0.shape[0]-self.frame_now_num, :, :] = self.img0[self.frame_now_num:, :, :]
            # black_img[self.frame_now_num:, :, :] = self.img1[:self.org_img0.shape[0]-self.frame_now_num, :, :]
            # img = self.img0[
            #       int(self.crop_start_y + self.frame_now_num + 2):int(self.crop_start_y + self.frame_now_num + 2 + 250),
            #       :, :]
            # img = find_sobel(img)
            # print(self.frame_now_num, _img_path)
            # if img.shape[0] != 250:
            if self.frame_now_num >= self.org_img0.shape[0]/3:
                1/0
            self.frame_now_num += 1
            return True, black_img
        except:
            print(f"批量图片读取遇到bug:{traceback.format_exc()}")
            return False, None


class App:
    def __init__(self, img_list, stitch_key):
        self.stitch_key = stitch_key
        self.track_len = 5
        self.detect_interval = 2  # 每5帧检测新的关键点，不延用上一帧的关键点做跟踪
        self.tracks = []
        self.cam = BatchImg(img_list)
        self.start_and_now_frame = 0
        self.all_frmae_num = self.cam.all_frame_num
        self.first_frame = self.cam.img1
        self.crop_para = 300  # 用于优化拼接速度的，从原图1/3位置往前的200行像素开始，到1/3位置往往后200行像素结束,这个参数可调
        # if int(self.first_frame.shape[0]/3 - self.crop_para) < 0:
        #     self.crop_para = int(self.first_frame.shape[0]/3)
        #     print("裁剪参数会超出视频帧的范围，把裁剪参数设为帧高的1/3")
        self.col_para = 2  # 3是从中间为开始，x坐标往前后抠1/3像素做匹配，最低为2
        self.first_up_half_img = self.first_frame  # [0: int(self.first_frame.shape[0]/3 - self.crop_para), :].copy()      # 用于最后拼在最上面
        self.width = self.first_frame.shape[1]
        self.add_result = self.first_frame  # [int(self.first_frame.shape[0]/3 - self.crop_para): int(self.first_frame.shape[0]/3 + self.crop_para), :].copy()
        # self.first_frame = self.first_frame[int(self.first_frame.shape[0]/3 - self.crop_para): int(self.first_frame.shape[0]/3 + self.crop_para),
        #                    int(self.width/2 - int(self.width/self.col_para)):int(self.width/2 + int(self.width/self.col_para))].copy()
        self.last_frame = None
        self.jump_frame = 2
        self.prev_frame = None

    def run(self, debug=False):
        print("开始拼接......")
        t_start = time.time()
        # while self.start_and_now_frame < self.end_frame:
        # cv2.namedWindow("img1", 0)
        # cv2.namedWindow("img2", 0)
        # cv2.resizeWindow("img1", 800, 800)
        # cv2.resizeWindow("img2", 800, 800)
        frame_list = []
        frame_diff_y_list = []
        collect_offset_y = []
        while self.start_and_now_frame < 6000:
            if self.start_and_now_frame == 0:
                org_frame = self.first_frame
                _ret = True
            else:
                _ret, org_frame = self.cam.read()
            if _ret:
                self.prev_frame = org_frame.copy()
            if self.start_and_now_frame == 0:
                frame_gray = cv2.cvtColor(org_frame, cv2.COLOR_BGR2GRAY)
                vis = org_frame.copy()
            if self.start_and_now_frame % self.jump_frame == 0 and self.start_and_now_frame > 0:
                t1 = time.time()
                if not _ret or self.start_and_now_frame >= 6000:
                    print("_ret:", _ret, " , self.start_and_now_frame:", self.start_and_now_frame)
                    print("视频流有问题，读不到图片了，这里拿前一帧作为最后一帧，拼到最后面去")
                    self.last_frame = self.prev_frame
                    print(" 开始拼最后一张 ")
                    crop_next_roi = self.last_frame[int(self.last_frame.shape[0] / 3) + self.crop_para:, :]
                    new_mask = np.zeros(
                        (self.add_result.shape[0] + crop_next_roi.shape[0], self.add_result.shape[1], 3), np.uint8)
                    new_mask[0:self.add_result.shape[0], :] = self.add_result
                    new_mask[self.add_result.shape[0]:, :] = crop_next_roi
                    self.add_result = new_mask.copy()
                    break

                # if self.start_and_now_frame == int(self.all_frmae_num - self.detect_interval - 2):
                #     self.last_frame = org_frame
                #     print("---- coming last frame -----")
                temp_crop_frame = org_frame
                frame = org_frame.copy()
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                vis = frame.copy()
                if len(self.tracks) > 0:  # and self.frame_idx % self.detect_interval == 0:
                    # img0, img1 = self.prev_gray, frame_gray
                    img0, img1 = cv2.cvtColor(self.first_frame, cv2.COLOR_BGR2GRAY), frame_gray
                    if debug:
                        cv2.imshow("img1", img0)
                        cv2.imshow("img2", img1)
                    # cv2.waitKey(0)
                    p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                    # 前一帧的角点和当前帧的图像作为输入来得到角点在当前帧的位置
                    p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                    # 当前帧跟踪到的角点及图像和前一帧的图像作为输入来找到前一帧的角点位置
                    p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)

                    # print("找角点一张耗时：", time.time() - t1)
                    d = abs(p0 - p0r).reshape(-1, 2).max(-1)  # 得到角点回溯与前一帧实际角点的位置变化关系
                    good = d < 1  # 判断d内的值是否小于1，大于1被认为是错误的跟踪点
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
                        cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                    # print("画点耗时：", time.time() - tc)
                    self.tracks = new_tracks

                    # 开始拼接：只用y变量；因为这个光流法的特征点检测和匹配还是很准的，直接采用前一帧和后一帧的y坐标插值作为移动的距离，直接拼
                    diff_y_list = []
                    # print("len(self.tracks):", len(self.tracks))
                    if len(self.tracks) >= 1:
                        for one_keyp in self.tracks:
                            one_diff_y = one_keyp[-2][1] - one_keyp[-1][1]
                            diff_y_list.append(one_diff_y)
                        # 优化1：对前2%的移动偏移量做个平均
                        sort_diff_y = sorted(diff_y_list, reverse=True)
                        try:  # 会出现前2%为0的情况，也就是特征点的数量很少，这里就直接拿前面5个的距离
                            cut_scale = int(np.rint(len(sort_diff_y) * 0.02))
                            if cut_scale >= 8:
                                sort_diff_y = sort_diff_y[:cut_scale]
                            else:
                                sort_diff_y = sort_diff_y[:]
                        except:
                            sort_diff_y = sort_diff_y[:5]
                        # if self.start_and_now_frame == 446:
                        #     print(sort_diff_y, len(sort_diff_y))
                        new_diff_y = int(sum(sort_diff_y) / len(sort_diff_y))  # 求前2%的y方向平均偏移
                        # diff_y = int(max(diff_y_list))    # 直接拿y方向的最大偏移
                        diff_y = new_diff_y
                        collect_offset_y.append(diff_y)
                        print(
                            f"key:{self.stitch_key}, self.all_frmae_num:{self.all_frmae_num}, self.start_and_now_frame:{self.start_and_now_frame}, "
                            f"diff_y:{diff_y}, send time:{time.time() - t_start}")
                        frame_list.append(self.start_and_now_frame)
                        frame_diff_y_list.append(diff_y)
                        if diff_y > 0 and self.last_frame is None:
                            crop_next_roi = temp_crop_frame[frame.shape[0] - diff_y:, :]
                            new_mask = np.zeros((self.add_result.shape[0] + diff_y, self.add_result.shape[1], 3),
                                                np.uint8)
                            new_mask[0:self.add_result.shape[0], :] = self.add_result
                            new_mask[self.add_result.shape[0]:, :] = crop_next_roi
                            self.add_result = new_mask.copy()
                        # 第一阶段优化：用前半张图片进行拼接，拼到最后一张时再用最后一张直接全部拼上去
                        if self.last_frame is not None:
                            print(" 开始拼最后一张 ")
                            crop_next_roi = self.last_frame[int(self.last_frame.shape[0] / 3) + self.crop_para:, :]
                            new_mask = np.zeros(
                                (self.add_result.shape[0] + crop_next_roi.shape[0], self.add_result.shape[1], 3),
                                np.uint8)
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
                p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)  # shi-Tomasi角点检测
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
                ch = cv2.waitKey(2)
                if ch == 27:
                    break

        # 新版，批量图片的拼接，找出开始相同位置的起始位置，那个位置就是整体的偏移量了
        print(f"offset_y:{collect_offset_y}")
        coor_list = []
        len_list = []
        pre_status = False
        lx = 0
        coor = 0
        for i, (frame_index, oy) in enumerate(zip(frame_list, frame_diff_y_list)):
            if oy < 0:
                coor = frame_index
                # print("y:", frame_index, oy)
                # break
        max_len_coor = coor# * self.jump_frame
        print(f"两张图片对比的偏移量:{max_len_coor}")

        org_img0 = self.cam.org_img0.copy()
        org_img1 = self.cam.org_img1.copy()
        # 把前半部分的也加上去
        new_mask = np.zeros((org_img0.shape[0] + max_len_coor, org_img0.shape[1], 3),
                            np.uint8)
        new_mask[0:org_img0.shape[0], :, :] = org_img0
        temp_crop_img = org_img1[int(org_img1.shape[0]-max_len_coor):, :, :].copy()
        new_mask[org_img0.shape[0]:, :] = temp_crop_img
        write_name = str(time.strftime("%Y_%m_%d-%H_%M_%S")) + ".png"

        cv2.imwrite(f"./cache/{write_name}", new_mask)
        print(f"保存拼接成功，路径：./cache/{write_name}, key:{self.stitch_key}")
        global all_crop_img
        all_crop_img[self.stitch_key] = temp_crop_img


def main(temp_img0, temp_img1, stitch_key):
    App([temp_img0, temp_img1], stitch_key).run(debug=False)
    print('Done')


def multi_process(img_list):
    # img0, img1, img2, img3, img4, img5, img6, img7, img8, img9 = img_list
    for i in range(len(img_list)-1):
        print("main for i :", i)
        p_app = Thread(target=main, args=[img_list[i], img_list[i+1], f"{i}_{i+1}"])
        p_app.start()
    global all_crop_img
    while True:
        if all_crop_img.keys().__len__() == len(img_list) - 1:
            key_list = sorted(all_crop_img.keys())
            print("key_list:", key_list)
            add_img = img_list[0]
            fscale = 0.3
            add_img = cv2.resize(add_img, (0, 0), fx=fscale, fy=fscale)
            for k in key_list:
                crop_img = all_crop_img[k]
                temp_add_img = np.zeros((add_img.shape[0] + crop_img.shape[0], add_img.shape[1], 3), dtype=np.uint8)
                temp_add_img[0:add_img.shape[0], :, :] = add_img
                temp_add_img[add_img.shape[0]:, :] = crop_img
                add_img = temp_add_img.copy()
                print(f"k :{k}, crop img shape:", crop_img.shape)

            write_name = "collect_" + str(time.strftime("%Y_%m_%d-%H_%M_%S")) + ".png"
            cv2.imwrite(f"./cache/{write_name}", add_img)
            print("开始拼接汇总 -------------")
            break
    return None


global all_crop_img
all_crop_img = {}

if __name__ == '__main__':
    t1 = time.time()
    img_l = []
    for i in sorted(os.listdir(r'F:\1_sheng\image_stitch\img2jpg\10_max')):
        img_path = os.path.join(r'F:\1_sheng\image_stitch\img2jpg\10_max', i)
        print(img_path)
        main_img = cv2.imread(img_path)
        img_l.append(main_img)
    multi_process(img_l)
    print("耗时：", time.time() - t1)
    cv2.destroyAllWindows()
