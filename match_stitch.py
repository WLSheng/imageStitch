import os
import numpy as np
import cv2
import time
import matplotlib
import matplotlib.pyplot as plt
import math
import traceback
from threading import Thread


def cv_imread(file_path):
    # 读取中文路径下的图片
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.int8), -1)
    if cv_img.shape[-1] == 4:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGBA2RGB)
    return cv_img


def match(org_test_img, template_img, thread_key):
    # org_test_img = cv_imread(r'F:\1_sheng\image_stitch\img2jpg\10_liang/0005.jpg')
    # org_test_img = cv2.bilateralFilter(org_test_img, 9, 75, 75)
    new_w = int(org_test_img.shape[1] * scale)
    new_h = int(org_test_img.shape[0] * scale)
    print("new_w, new_h: ", new_w, new_h)
    org_test_img = cv2.resize(org_test_img, (new_w, new_h))
    print("img.shape:", org_test_img.shape)

    # template_img = cv_imread(r'F:\1_sheng\image_stitch\img2jpg\10_liang/0004.jpg')
    # template_img = cv2.bilateralFilter(template_img, 9, 75, 75)
    template_img = cv2.resize(template_img, (new_w, new_h))
    # org_template_img = template_img[int(new_h / 3):, 50:int(template_img.shape[1]*4/5), :]
    org_template_img = template_img[int(new_h / 4):, 5:, :]
    # print("org_template_img.shape:", org_template_img.shape)

    # methods = ['cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    methods = ['cv2.TM_CCOEFF_NORMED']
    w, h, _ = org_template_img.shape
    for meth in methods:
        img = org_test_img.copy()
        method = eval(meth)
        # 应用模板匹配
        # cv2.imshow("img", img)
        # cv2.imshow("org_template_img", org_template_img)
        # cv2.waitKey(3)
        res = cv2.matchTemplate(img, org_template_img, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # 如果方法是TM_SQDIFF或TM_SQDIFF_NORMED，则取最小值
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + h, top_left[1] + w)
        # print("1111:", img.shape, top_left, bottom_right)
        # cv2.rectangle(img, top_left, bottom_right, 255, 4)
        # offset_y = top_left[1] + h
        # 映射回大图的尺寸
        offset_y = int(new_h / 4 - top_left[1])
        print("偏移量：", offset_y, top_left[1] + h)
        crop_img = org_test_img[int(org_test_img.shape[0] - offset_y):, :, ]
        all_crop_img[thread_key] = crop_img

        temp_add_img = np.zeros((template_img.shape[0] + crop_img.shape[0], template_img.shape[1], 3), dtype=np.uint8)
        temp_add_img[0:template_img.shape[0], :, :] = template_img
        temp_add_img[org_test_img.shape[0]:, :] = crop_img
        # print("crop.shape:", crop_img.shape)
        # cv2.imshow("template_img", template_img)
        # cv2.waitKey(1)
        # if crop_img.shape[0] > 1:
        #     cv2.imshow("crop", crop_img)
        # else:
        #     print("模板匹配有问题，大bug  ")
        # cv2.waitKey(3)
        # plt.subplot(121)
        # plt.imshow(res, cmap='gray')
        # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        # plt.subplot(122)
        # plt.imshow(img, cmap='gray')
        # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        # plt.suptitle(meth)
        #
        # plt.figure(999)
        # temp_add_img = cv2.cvtColor(temp_add_img, cv2.COLOR_BGR2RGB)
        # plt.imshow(temp_add_img)
        #
        # plt.show()

    # # 多对象的模板匹配
    # w, h, _ = org_template_img.shape
    # res = cv2.matchTemplate(img, org_template_img, cv2.TM_CCOEFF_NORMED)
    # threshold = 0.8
    # loc = np.where(res >= threshold)
    # for pt in zip(*loc[::-1]):
    #     cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    # cv2.imshow('res', res)
    # cv2.waitKey()
    print(f"thread_key:{thread_key}, over ")


def multi_process(img_list):
    # img0, img1, img2, img3, img4, img5, img6, img7, img8, img9 = img_list
    for i in range(len(img_list)-1):
        print("main for i :", i, f", key: {str(i).zfill(2)}{str(i+1).zfill(2)}")
        p_app = Thread(target=match, args=[img_list[i+1], img_list[i], f"{str(i).zfill(2)}{str(i+1).zfill(2)}"])
        p_app.start()
    global all_crop_img
    while True:
        if all_crop_img.keys().__len__() == len(img_list) - 1:
            key_list = sorted(all_crop_img.keys())
            print("key_list:", key_list)
            add_img = img_list[0]
            add_img = cv2.resize(add_img, (0, 0), fx=scale, fy=scale)
            for k in key_list:
                crop_img = all_crop_img[k]
                temp_add_img = np.zeros((add_img.shape[0] + crop_img.shape[0], add_img.shape[1], 3), dtype=np.uint8)
                temp_add_img[0:add_img.shape[0], :, :] = add_img
                # print(f"crop_img.shape:{crop_img.shape}, temp_add_img.shape:{temp_add_img.shape}")
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
scale = 0.3

if __name__ == '__main__':
    t1 = time.time()
    img_l = []
    folder = r'F:\1_sheng\image_stitch\img2jpg\10_2'
    for i, name in enumerate(sorted(os.listdir(folder), reverse=False)):
        img_path = os.path.join(folder, name)
        print(i, img_path)
        # main_img = find_sobel(cv2.imread(img_path))
        main_img = cv2.imread(img_path)
        # main_img = cv2.resize(main_img, (1920, 1080))
        img_l.append(main_img)
    multi_process(img_l)
    print("耗时：", time.time() - t1)
    cv2.destroyAllWindows()
