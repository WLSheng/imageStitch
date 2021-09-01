import os
import numpy as np
import cv2
import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics.pairwise import cosine_similarity, paired_distances
import math


def cv_imread(file_path):
    # 读取中文路径下的图片
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.int8), -1)
    if cv_img.shape[-1] == 4:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGBA2RGB)
    return cv_img


def show_gradient(two_gradient, two_org_img=None):

    show = True
    # show = False
    if show:
        plt.figure(1)
        for i in range(2):
            plt.subplot(2, 2, 1+i*2)
            # plt.axis('off')
            plt.imshow(two_gradient[i][:, :])
            # plt.subplot(2, 4, 2+i*4)
            # # plt.axis('off')
            # plt.imshow(two_gradient[i][:, :, 1])
            # plt.subplot(2, 4, 3+i*4)
            # # plt.axis('off')
            # plt.imshow(np.max(two_gradient[i], 2))
            # plt.axis('off')
            if two_org_img:
                plt.subplot(2, 2, 2+i*2)
                plt.title("org img")
                plt.imshow(two_org_img[i])
            # else:
            #     plt.imshow(np.max(two_gradient[i], 2))
            #     plt.subplot(2, 4, 4+i*4)
        plt.pause(0.5)
        # plt.show()


def find_gradient(find_gradient_img):

    find_gradient_img = cv2.cvtColor(find_gradient_img, cv2.COLOR_BGR2GRAY)
    find_gradient_img = cv2.GaussianBlur(find_gradient_img, (3, 3), 3)
    # cv2.imshow("gs", find_gradient_img)
    # canny = cv2.Canny(find_gradient_img, 10, 150)
    # cv2.imshow("canny", canny)
    # cv2.waitKey(3)
    row, column = find_gradient_img.shape
    find_gradient_img = np.asarray(find_gradient_img, dtype=np.float)
    gradient = np.zeros((row, column, 2), dtype=np.float)

    # 有一个加速梯度计算的优化思路：对x方向计算梯度时，把原图往右挪一个像素，这样再把两张图片相减，直接得到x方向的梯度，然后两个梯度图再合并
    for x in range(row - 1):
        for y in range(column - 1):
            gx = find_gradient_img[x + 1, y] - find_gradient_img[x, y]
            gy = find_gradient_img[x, y + 1] - find_gradient_img[x, y]
            gradient[x, y] = (gx, gy)

    # 后面看看匹配时需不需要把小于0的梯度向量置0
    gradient[gradient < 0] = 0

    return gradient


global figure
figure = 99


def find_sobel_gradient(t_org_img):
    global figure
    t_org_img = cv2.GaussianBlur(t_org_img, (7, 7), 0, sigmaY=0, borderType=cv2.BORDER_REPLICATE)
    # t1 = time.time()
    # 这里是转灰度图的梯度
    # t_org_img = cv2.cvtColor(t_org_img, cv2.COLOR_BGR2GRAY)
    # # t_org_img = t_org_img.astype(np.float)
    # r_gx = cv2.Sobel(t_org_img, cv2.CV_32F, 1, 0, ksize=3, scale=1.0, delta=0.0, borderType=cv2.BORDER_REPLICATE)
    # r_gy = cv2.Sobel(t_org_img, cv2.CV_32F, 0, 1, ksize=3, scale=1.0, delta=0.0, borderType=cv2.BORDER_REPLICATE)
    # r_gxy = np.maximum(r_gx, r_gy)
    # plt.subplot(1, 5, 1)
    # plt.imshow(r_gx)
    # plt.subplot(1, 5, 2)
    # plt.imshow(r_gy)
    # plt.subplot(1, 5, 3)
    # plt.imshow(r_gxy)
    # plt.show()

    # 对三通道的原图各自求梯度图
    sobel_type = cv2.CV_64F
    sobel_size = 2
    t_org_img = t_org_img.astype(np.float64)
    r, g, b = cv2.split(t_org_img)
    r_gx = cv2.Sobel(r, sobel_type, sobel_size, 0, ksize=3, scale=1.0, delta=0.0, borderType=cv2.BORDER_REPLICATE)
    r_gy = cv2.Sobel(r, sobel_type, 0, sobel_size, ksize=3, scale=1.0, delta=0.0, borderType=cv2.BORDER_REPLICATE)
    # r_gxy = np.maximum(r_gx, r_gy)

    g_gx = cv2.Sobel(g, sobel_type, sobel_size, 0, ksize=3, scale=1.0, delta=0.0, borderType=cv2.BORDER_REPLICATE)
    g_gy = cv2.Sobel(g, sobel_type, 0, sobel_size, ksize=3, scale=1.0, delta=0.0, borderType=cv2.BORDER_REPLICATE)
    # g_gxy = np.maximum(g_gx, g_gy)

    b_gx = cv2.Sobel(b, sobel_type, sobel_size, 0, ksize=3, scale=1.0, delta=0.0, borderType=cv2.BORDER_REPLICATE)
    b_gy = cv2.Sobel(b, sobel_type, 0, sobel_size, ksize=3, scale=1.0, delta=0.0, borderType=cv2.BORDER_REPLICATE)
    # b_gxy = np.maximum(b_gx, b_gy)

    # rg_gxy = np.maximum(r_gxy, g_gxy)
    # rgb_gxy = np.maximum(rg_gxy, b_gxy)

    # 怀疑计算角度弄错了，在梯度出来后就错了一步，梯度方向是有专门的函数计算的，
    # rg_gx = np.maximum(r_gx, g_gx)
    # rgb_gx = np.maximum(rg_gx, b_gx)
    # rg_gy = np.maximum(r_gy, g_gy)
    # rgb_gy = np.maximum(rg_gy, b_gy)
    # rgb_gxy = np.maximum(rgb_gx, rgb_gy)
    rg_gx = r_gx + g_gx
    rgb_gx = rg_gx + b_gx
    rg_gy = r_gy + g_gy
    rgb_gy = rg_gy + b_gy
    rgb_gxy = rgb_gx + rgb_gy

    # plt.figure(figure)
    # plt.imshow(rgb_gxy)
    # plt.show()
    # rgb_gx[rgb_gx < 0] = 0
    # rgb_gy[rgb_gy < 0] = 0
    magnitude = rgb_gx * rgb_gx + rgb_gy * rgb_gy
    # rgb_gx = cv2.blur(rgb_gx, (6, 6))
    # rgb_gy = cv2.blur(rgb_gy,  (6, 6))
    # rgb_gx = rgb_gx * rgb_gy
    # rgb_gy = rgb_gy * rgb_gx
    # rgb_gx = cv2.GaussianBlur(rgb_gx, (3, 3), 0, sigmaY=0, borderType=cv2.BORDER_REPLICATE)
    # rgb_gy = cv2.GaussianBlur(rgb_gy, (3, 3), 0, sigmaY=0, borderType=cv2.BORDER_REPLICATE)
    rgb_gx = rgb_gx.astype(np.float64)
    rgb_gy = rgb_gy.astype(np.float64)
    # rgb_gx = abs(rgb_gx)#.astype(np.float64)
    # rgb_gy = abs(rgb_gy)#.astype(np.float64)
    # rgb_gx[rgb_gx < 100] = -1
    # rgb_gy[rgb_gy < 100] = -1
    _ori_x = cv2.phase(rgb_gx, rgb_gy, angleInDegrees=True)
    _ori_y = cv2.phase(rgb_gy, rgb_gx, angleInDegrees=True)

    direction = np.arctan(rgb_gy/rgb_gx)
    print(direction)
    direction1 = np.arctan(rgb_gx/rgb_gy)
    # print(direction1)
    # direction2 = np.cos(direction) * 180 / np.pi
    direction2 = direction1 * 180 / np.pi
    # print(direction2)
    # plt.figure(111)
    # plt.subplot(1, 2, 1)
    # plt.imshow(direction)
    # plt.subplot(1, 2, 2)
    # plt.imshow(direction2)
    # plt.show()

    orientation = np.maximum(_ori_y, _ori_x)
    # orientation = _ori_x
    print(rgb_gxy.shape, orientation.shape)
    plt.figure(11111)
    plt.subplot(1, 5, 1)
    plt.imshow(_ori_x)
    plt.subplot(1, 5, 2)
    plt.imshow(_ori_y)
    plt.subplot(1, 5, 3)
    plt.imshow(rgb_gxy)
    plt.subplot(1, 5, 4)
    plt.imshow(orientation)
    plt.subplot(1, 5, 5)
    plt.imshow(direction)
    plt.show()
    # _temp_gx = abs(np.cos(rgb_gx))
    # _temp_gy = abs(np.cos(rgb_gy))
    # # _temp_gxy = np.cos(rgb_gxy)
    # _temp_gxy = np.maximum(_temp_gx, _temp_gy)
    # print(_temp.shape)
    plt.figure(12)
    plt.subplot(1, 5, 1)
    plt.imshow(rgb_gx)
    plt.subplot(1, 5, 2)
    plt.imshow(rgb_gy)
    plt.subplot(1, 5, 3)
    plt.imshow(rgb_gxy)
    plt.subplot(1, 5, 4)
    plt.imshow(orientation)
    plt.show()

    # _ori = np.zeros(orientation.shape, dtype=np.float64)
    # _ori[4:, 4:] = orientation[:orientation.shape[0]-4, :orientation.shape[1]-4]
    # orientation = _ori.copy()

    # 拿到第一步的梯度图后，进行3X3放大范围找梯度，拿最大的，根据梯度强度图来重新定义梯度方向图，去掉梯度强度低的错误方向，排除干扰
    new_gradient = np.zeros(rgb_gxy.shape, dtype=np.float)
    new_ori = np.zeros(orientation.shape, dtype=np.float)
    # new_ori = orientation
    bin_ori = np.zeros(orientation.shape, dtype=np.float)
    endx = rgb_gxy.shape[1]
    endy = rgb_gxy.shape[0]
    offset = 1
    gradient_threshold = 10     # 论文说的卡一个比较小的梯度阈值
    qual_space = 180 / 8  # 论文里说分8等份
    # 第一次循环，把梯度值用3X3放大些,后来分放弃这个思路，感觉会影响后面的步骤
    for x in range(endx):
        for y in range(endy):
            # min_x = max(0, x-offset)     # 这里是3X3的
            # min_y = max(0, y-offset)
            # max_x = min(x+offset, endx)
            # max_y = min(y+offset, endy)
            # find_max_gradient_img = rgb_gxy[min_y:max_y, min_x:max_x]
            # max_gradient_value = np.max(find_max_gradient_img)
            # find_max_ori_img = orientation[min_y:max_y, min_x:max_x]
            # max_ori_value = np.max(find_max_ori_img)
            # new_gradient[y, x] = rgb_gxy[y, x]
            if rgb_gxy[y, x] > gradient_threshold:
                # new_gradient[y, x] = max_gradient_value
                new_ori[y, x] = orientation[y, x]
            else:
                # new_gradient[y, x] = 0
                new_ori[y, x] = 0
            if new_ori[y, x] <= 180.0:
                ori = min(math.floor(new_ori[y, x] / qual_space), 7)     # 计算当前的梯度方向是哪个区域的，
                bin_ori[y, x] = ori
            else:
                ori = min(math.floor((360-new_ori[y, x]) / qual_space), 7)     # 计算当前的梯度方向是哪个区域的，
                bin_ori[y, x] = ori

    # 开始制作广播方向，未完待续 2021.08.07
    offset = 4      # 在8X8的范围内统计广播方向
    for x in range(endx):
        for y in range(endy):
            temp_spread = ['0', '0', '0', '0', '0', '0', '0', '0']
            min_x = max(0, x-offset)     # 这里是3X3的
            min_y = max(0, y-offset)
            max_x = min(x+offset, endx)
            max_y = min(y+offset, endy)
            count_ori_img = bin_ori[min_y:max_y, min_x:max_x]
            temp_count_ori = count_ori_img.flatten()
            # print("read ori:", temp_count_ori)
            # print("set: ", set(temp_count_ori))     # set:  {0.0, 1.0, 4.0, 7.0}
            for o in set(temp_count_ori):
                temp_spread[int(o)] = '1'
            pix_binary = ''.join(temp_spread)[::-1]
            # 拿到方向量化后就计算左、上方向的响应图

    # 这样就拿到了两个图，一个是方向图，一个是梯度强度图
    # gx = cv2.cvtColor(gx, cv2.COLOR_BGR2GRAY)
    # gy = cv2.cvtColor(gy, cv2.COLOR_BGR2GRAY)
    # rgb_gxy = np.where(rgb_gxy < 0, 0, np.where(rgb_gxy > 255, 255, rgb_gxy))
    plt.figure(222)
    plt.subplot(1, 5, 1)
    plt.imshow(rgb_gxy)
    plt.subplot(1, 5, 2)
    plt.imshow(new_gradient)
    plt.subplot(1, 5, 3)
    plt.imshow(orientation)
    plt.figure(4)
    plt.imshow(new_ori)
    # plt.subplot(1, 5, 5)
    plt.figure(6)
    plt.imshow(bin_ori)
    plt.show()
    # plt.pause(0.5)
    figure += 1
    return new_gradient


def gradient_match():
    t1 = time.time()
    org_img = cv_imread('./gradient_match/toy_duck_1.jpg')
    # org_img = cv_imread('./gradient_match/640.jpg')
    org_img = cv_imread('./gradient_match/left_up_ans_template.jpg')
    org_img = cv_imread('./gradient_match/test_gradient_match_1.png')
    # org_img = cv_imread('./gradient_match/test_gradient_match_0.png')
    # org_img = cv_imread('./gradient_match/yuan.png')

    # org_gradient = find_gradient(org_img)
    # template_gradient = find_gradient(template_img)
    t1 = time.time()
    org_gradient = find_sobel_gradient(org_img)
    t2 = time.time()
    print("找sobel梯度耗时：", t2-t1)
    # cos_gradient, ori_gradient = gradient2angle(org_gradient)
    # t3 = time.time()
    # print("梯度转方向量化耗时：", t3-t2)
    # spread_orientation(ori_gradient)
    # t4 = time.time()
    # print("方向量化转广播耗时：", t4-t3)
    # template_gradient = find_sobel_gradient(template_img)


def find_gradient_orientation(img):
    """
    思路：
    sobel -> sqrt(gx^2 + gy^2) -> 算出幅值 -> 根据幅值（模），把小幅值去掉，整理出一份新的gx,gy图，然后在新的gx,gy图上逐像素归一化到1，归一化后直接拿去算相似度？

    :return:
    """
    t_org_img = cv2.GaussianBlur(img, (5, 5), 0, sigmaY=0, borderType=cv2.BORDER_REPLICATE)
    sobel_type = cv2.CV_64F
    sobel_size = 1  # 一/二阶导数
    t_s = time.time()
    t_org_img = t_org_img.astype(np.float64)
    gradient_is_rgb = True
    sobel_ksize = 7
    #  ------------- 对三通道的原图各自求梯度图  ----------------
    if gradient_is_rgb:
        r, g, b = cv2.split(t_org_img)
        r_gx = cv2.Sobel(r, sobel_type, sobel_size, 0, ksize=sobel_ksize, scale=1.0, delta=1.0, borderType=cv2.BORDER_REPLICATE)
        r_gy = cv2.Sobel(r, sobel_type, 0, sobel_size, ksize=sobel_ksize, scale=1.0, delta=1.0, borderType=cv2.BORDER_REPLICATE)

        g_gx = cv2.Sobel(g, sobel_type, sobel_size, 0, ksize=sobel_ksize, scale=1.0, delta=1.0, borderType=cv2.BORDER_REPLICATE)
        g_gy = cv2.Sobel(g, sobel_type, 0, sobel_size, ksize=sobel_ksize, scale=1.0, delta=1.0, borderType=cv2.BORDER_REPLICATE)

        b_gx = cv2.Sobel(b, sobel_type, sobel_size, 0, ksize=sobel_ksize, scale=1.0, delta=1.0, borderType=cv2.BORDER_REPLICATE)
        b_gy = cv2.Sobel(b, sobel_type, 0, sobel_size, ksize=sobel_ksize, scale=1.0, delta=1.0, borderType=cv2.BORDER_REPLICATE)

        rg_gx = np.maximum(r_gx, g_gx)
        rgb_gx = np.maximum(rg_gx, b_gx)
        rg_gy = np.maximum(r_gy, g_gy)
        rgb_gy = np.maximum(rg_gy, b_gy)
        _rgb_gx = np.expand_dims(rgb_gx, axis=-1)
        _rgb_gy = np.expand_dims(rgb_gy, axis=-1)
        rgb_gxy = np.concatenate([_rgb_gx, _rgb_gy], axis=-1)
    else:
        # 灰度图求梯度
        t_org_img = cv2.cvtColor(t_org_img, cv2.COLOR_BGR2GRAY)
        rgb_gx = cv2.Sobel(t_org_img, sobel_type, sobel_size, 0, ksize=sobel_ksize, scale=1.0, delta=0.0, borderType=cv2.BORDER_REPLICATE)
        rgb_gy = cv2.Sobel(t_org_img, sobel_type, 0, sobel_size, ksize=sobel_ksize, scale=1.0, delta=0.0, borderType=cv2.BORDER_REPLICATE)

        _rgb_gx = np.expand_dims(rgb_gx, axis=-1)
        _rgb_gy = np.expand_dims(rgb_gy, axis=-1)
        rgb_gxy = np.concatenate([_rgb_gx, _rgb_gy], axis=-1)

    # plt.figure(1212)
    # plt.subplot(1, 2, 1)
    # plt.imshow(rgb_gxy[:, :, 0])
    # plt.subplot(1, 2, 2)
    # plt.imshow(rgb_gxy[:, :, 1])
    # plt.show()
    print("sobel 耗时:", time.time() - t_s)

    t1 = time.time()
    # ---------- 算幅值 ， 然后卡幅值 ， 得出一个新的gx,gy   -------------
    norm = np.sqrt(rgb_gx * rgb_gx + rgb_gy * rgb_gy)
    gradient_threshold = np.max(norm) * 0.01
    print("gradient_threshold: ", gradient_threshold)
    norm[norm < gradient_threshold] = 0         # todo : 卡的梯度的幅值参数

    # new_gxy = np.zeros(rgb_gxy.shape, dtype=np.float)  # 过滤小幅值后的向量
    # gxy_norm = np.zeros(rgb_gxy.shape, dtype=np.float)  # 过滤后的向量的归一化，还是一个向量
    gxy_orientation = np.zeros((rgb_gxy.shape[0], rgb_gxy.shape[1]), dtype=np.float)  # 向量的方向，没用到

    # 下面是第一种方法，把小于一定模值的像素的gx，gy去掉，用numpy的维度乘法
    temp_norm = norm.copy()
    temp_norm[temp_norm > 0] = 1
    temp_norm = np.expand_dims(temp_norm, axis=-1)
    new_gxy = temp_norm * rgb_gxy   # 去掉了梯度幅值小于一定值的方向向量
    _norm = np.expand_dims(norm, axis=-1)
    # new_gxy[new_gxy == -0] = 0.
    print("取模、归一化耗时：", time.time() - t1)
    gxy_norm = new_gxy / _norm      # 向量归一化
    gxy_norm = np.where(np.isnan(gxy_norm), 0, gxy_norm)

    # 下面是第二种方法，把小于一定模值的像素的gx，gy去掉，两个for循环，顺便把两个向量归一化
    # endx = norm.shape[1]
    # endy = norm.shape[0]
    # for x in range(endx):
    #     for y in range(endy):
    #         if norm[y, x] > 0:
    #             _temp = rgb_gxy[y, x]
    #             # print("_temp: ", _temp)
    #             new_gxy[y, x] = _temp
    #             softmax = _temp / norm[y, x]
    #             gxy_norm[y, x] = softmax

    return gxy_norm, norm, gxy_orientation, gradient_threshold   # 第一个是xy方向向量归一化后的向量，第二个是原始xy方向向量的模，0代表没有梯度，或梯度很小被过滤了，第三个是方向


def gradient_match_baseline():
    """
    思路：
    sobel -> sqrt(gx^2 + gy^2) -> 算出幅值 -> 根据幅值（模），把小幅值去掉，整理出一份新的gx,gy图，然后在新的gx,gy图上逐像素归一化到1，归一化后直接拿去算相似度？

    :return:
    """
    # template_img = np.rot90(template_img, -2)
    org_test_img = cv_imread(r'F:\1_sheng\image_stitch\img2jpg\5_liang/1043,500000.jpg')
    test_img = cv2.resize(org_test_img, (512, 1024))
    # test_img = cv_imread(r'F:\1_sheng\card\06_test\0625-0627/2021-06-27_08_51_37.jpg')
    coor_type = 1      # 1 为左上角，2为右上角，3为左下角，4为右下角

    org_template_img = cv_imread(r'F:\1_sheng\image_stitch\img2jpg\5_liang/1418,500000.jpg')
    template_img = cv2.resize(org_template_img, (512, 1024))
    template_img = template_img[512:, 10:, :]
    # template_img = test_img
    # template_img = np.rot90(template_img, -3)
    # cv2.imwrite('./gradient_match/templates/right_lower_ans_template_2.jpg', template_img)
    # gaotong_gradient = find_gaotong_gradient(test_img)
    # exit()
    # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(5, 5))
    # test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    # clahe.apply(test_img, test_img)
    # test_img = cv2.cvtColor(test_img, cv2.COLOR_GRAY2RGB)

    t_all = time.time()
    test_gxy_norm_vector, test_gradient_norm, test_ori, test_threshold_gradient = find_gradient_orientation(test_img)
    template_gxy_norm_vector, template_gradient_norm, template_ori, template_threshold_gradient = find_gradient_orientation(template_img)

    plt.figure(121)
    plt.subplot(1, 2, 1)
    plt.imshow(test_gradient_norm)
    plt.subplot(1, 2, 2)
    plt.imshow(template_gradient_norm)
    # plt.figure(1222)
    # plt.subplot(1, 3, 1)
    # plt.imshow(test_gxy_norm_vector[:, :, 0])
    # plt.subplot(1, 3, 2)
    # plt.imshow(test_gxy_norm_vector[:, :, 1])
    # angle = test_gxy_norm_vector[:, :, 1] / test_gxy_norm_vector[:, :, 0]
    # plt.subplot(1, 3, 3)
    # plt.imshow(angle)
    # plt.show()
    plt.pause(0.2)

    end_y, end_x, _ = test_gxy_norm_vector.shape
    temp_y, temp_x, _ = template_gxy_norm_vector.shape
    match_score = np.zeros((test_gxy_norm_vector.shape[0], test_gxy_norm_vector.shape[1]), dtype=np.float)
    ts = time.time()
    count = len(np.where((template_gxy_norm_vector * template_gxy_norm_vector).sum(axis=2) > 0.5)[0])
    # print("count:", count, time.time() - ts)
    temp_gxy_vector = template_gxy_norm_vector.flatten()
    temp_gradient_vector = template_gradient_norm.flatten()
    # template_gradient_norm[template_gradient_norm > 0] = 1
    # print(template_gxy_norm_vector.shape, template_gradient_norm.shape, np.expand_dims(template_gradient_norm, axis=-1).shape)
    # template_gxy_gradient = np.concatenate([template_gxy_norm_vector, np.expand_dims(template_gradient_norm, axis=-1)], axis=-1)
    i = 0
    tsi = time.time()
    new_threshold = template_threshold_gradient * 2
    for y in range(0, end_y-temp_y, 1):
        for x in range(0, end_x-temp_x, 1):
            if test_gradient_norm[y, x] > new_threshold:
                i += 1
                org_temp_x = x + temp_x
                org_temp_y = y + temp_y
                test_crop_norm_vector = test_gxy_norm_vector[y:org_temp_y, x:org_temp_x]

                # numpy 用点积求相似度
                # test_vector = test_crop_norm_vector.flatten()
                # simi = np.dot(test_vector, temp_gxy_vector) / count
                # test_vector = test_gxy_gradient.flatten()
                # template_vector = template_gxy_gradient.flatten()
                # simi = np.dot(test_vector, template_vector) / count
                # 手写相似度
                simi = (test_crop_norm_vector*template_gxy_norm_vector).sum() / count     # 这是归一化置信度
                # print(simi)
                # test_crop_norm_vector[test_crop_norm_vector > 0] = 1
                # simi = (test_vector*template_vector).sum() / count     # 这是归一化置信度

                match_score[y, x] = simi
    print(i, 'time similarity:', time.time() - tsi)
    print("相似度计算耗时：", time.time() - ts, ", 总耗时： ", time.time() - t_all)
    max_score = np.max(match_score)
    coor = np.where(match_score == max_score)
    print("max_score: ", max_score, coor)
    # 下面是拼接的代码
    offset_y = 512 - coor[0]
    # 映射回大图的尺寸
    y = int(org_test_img.shape[0] * offset_y / 512)
    print(org_test_img.shape[0], y, offset_y)
    crop_img = org_test_img[int(org_test_img.shape[0]-y):, :, ]

    temp_add_img = np.zeros((org_test_img.shape[0] + crop_img.shape[0], org_test_img.shape[1], 3), dtype=np.uint8)
    temp_add_img[0:org_test_img.shape[0], :, :] = org_template_img
    temp_add_img[org_test_img.shape[0]:, :] = crop_img


    offset = 5
    if coor_type == 1:
        cv2.circle(test_img, (coor[1]+offset, coor[0]+offset), 2, [0, 255, 0], 2)
    elif coor_type == 2:
        cv2.circle(test_img, (coor[1]+template_img.shape[1]-offset, coor[0]+offset), 2, [0, 255, 0], 2)
    elif coor_type == 3:
        cv2.circle(test_img, (coor[1]+offset, coor[0]+template_img.shape[0]-offset), 2, [0, 255, 0], 2)
    elif coor_type == 4:
        cv2.circle(test_img, (coor[1]+template_img.shape[1]-offset, coor[0]+template_img.shape[0]-offset), 2, [0, 255, 0], 2)
    plt.figure(998)
    plt.imshow(match_score)
    plt.figure(110)
    plt.imshow(test_img)
    plt.figure(1111)
    plt.imshow(temp_add_img)
    plt.show()


if __name__ == '__main__':
    # gradient_match()
    gradient_match_baseline()
