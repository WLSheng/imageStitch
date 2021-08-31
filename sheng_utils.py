import time, os
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


def video2jpg():
    import imageio
    # filename="person15_walking_d1_uncomp.avi"
    # vid = imageio.get_reader(video_path,  'ffmpeg')
    # # number of frames in video
    # num_frames=vid._meta['nframes']
    # print(num_frames)
    t_start = time.time()
    video_path = r'F:\1_sheng\image_stitch\0630视频\c21ce7ae-481a-4246-b401-feb07a00fce8_193659.mp4'

    cap = cv2.VideoCapture(video_path)
    # all_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # fps = cap.get(cv2.CAP_PROP)
    all_frame = 0
    while True:
        _ret, org_frame = cap.read()
        if _ret:
            all_frame += 1
            # print(all_frame)
        else:
            break
    cap.release()
    cap = cv2.VideoCapture(video_path)
    save_all_frame = [50]
    for one in save_all_frame:
        save_frame = 0
        interval_frame = int(all_frame/one)
        while True:
            _ret, org_frame = cap.read()
            if _ret:
                if save_frame % interval_frame == 0:
                    save_path = rf'F:\1_sheng\image_stitch\img2jpg\{one}\\{str(save_frame).zfill(4)}.jpg'
                    cv2.imwrite(save_path, org_frame)
                    print(save_path)
                save_frame += 1
            else:
                break
        print(f'{one}, all frame: {all_frame}, send time:{time.time()-t_start}')


def find_sobel(gray):
    sobel_type = cv2.CV_64F
    sobel_size = 1  # 一/二阶导数
    sobel_ksize = 7
    rgb_gx = cv2.Sobel(gray, sobel_type, sobel_size, 0, ksize=sobel_ksize, scale=1.0, delta=0.0, borderType=cv2.BORDER_REPLICATE)
    rgb_gy = cv2.Sobel(gray, sobel_type, 0, sobel_size, ksize=sobel_ksize, scale=1.0, delta=0.0, borderType=cv2.BORDER_REPLICATE)

    _rgb_gx = np.expand_dims(rgb_gx, axis=-1)
    _rgb_gy = np.expand_dims(rgb_gy, axis=-1)
    rgb_gxy = abs(np.maximum(rgb_gx, rgb_gy))
    rgb_gxy = rgb_gxy / np.max(rgb_gxy) * 255.0
    rgb_gxy = np.asarray(rgb_gxy, dtype=np.uint8)
    # plt.imshow(rgb_gxy)
    # plt.show()
    return rgb_gxy


def two_diff():
    cv2.namedWindow("diff", 0)
    cv2.resizeWindow("diff", 800, 800)
    org_img0 = cv2.imread(r'F:\1_sheng\image_stitch\img2jpg\30\0042.jpg', 0)
    org_img1 = cv2.imread(r'F:\1_sheng\image_stitch\img2jpg\30\0056.jpg', 0)
    # rgb_gxy = np.concatenate([_rgb_gx, _rgb_gy], axis=-1)
    # plt.imshow(rgb_gx)
    # plt.show()
    img0 = find_sobel(org_img0)
    img1 = find_sobel(org_img1)
    # plt.imshow(rgb_gxy)
    # plt.show()
    cv2.imshow("diff", img1)
    cv2.waitKey(0)

    max_crop = 90
    crop0 = img0[int(img0.shape[0]*3/4):, :].copy()
    diff_list = []
    for i in range(max_crop):
        # 采用滑窗的方法，从下一张中抠下小图片
        start_y = int(img0.shape[0]*2/3 + i)
        end_y = int(start_y + crop0.shape[0])
        crop1 = img1[start_y:end_y, :].copy()
        # print(i, crop0.shape, crop1.shape)
        diff = abs(crop1 - crop0)
        diff_sum = np.sum(diff)
        # print(diff_sum)
        diff_list.append(diff_sum)
    diff_list = np.asarray(diff_list)
    # min_diff = np.min(diff_list)
    min_index = np.argmin(diff_list)
    line_y = int(img0.shape[0]*2/3 + min_index)
    line_yy = int(img0.shape[0]*2/3 + min_index + crop0.shape[0])
    cv2.line(org_img1, (0, line_y), (1000, line_y), color=[255], thickness=5)
    cv2.line(org_img1, (0, line_yy), (1000, line_yy), color=[255], thickness=5)
    print("sum_diff:", diff_list)
    cv2.imshow("diff", org_img1)
    cv2.waitKey(0)


def find_cycle(small_img):
    # show_crop_img = small_img.copy()
    one_row = np.asarray([np.sum(small_img, axis=1)/small_img.shape[1]])   # (5472, 500)
    # print(one_row, one_row.shape)
    # print(len(one_row), one_row.size, (draw_row, 0), (draw_row, img1.shape[0]))
    data = np.asarray(one_row, dtype=np.float).transpose(1, 0)
    transformed = np.fft.fft(data)
    return transformed


def test_fft_2():
    cv2.namedWindow("ttfft", 0)
    cv2.resizeWindow("ttfft", 1200, 1200)
    draw_row = 100
    crop_x = 4000
    img0 = cv2.imread(r"F:\1_sheng\image_stitch\img2jpg\10_max/1850,300000.jpg", 0)
    cv2.imshow("ttfft", img0)
    cv2.waitKey(99)
    one_row = img0[:, draw_row:draw_row+crop_x].copy()
    print(one_row.shape)
    img0_transformed = find_cycle(one_row)
    img1 = cv2.imread(r"F:\1_sheng\image_stitch\img2jpg\10_max/1760,350000.jpg", 0)
    one_row = img1[:, draw_row:draw_row+crop_x].copy()
    img1_transformed = find_cycle(one_row)

    print("start find cycle")
    plt.subplot(2, 1, 1)
    plt.plot(img0_transformed)
    plt.subplot(2, 1, 2)
    plt.plot(img1_transformed)
    plt.show()
    # np_trans = transformed.astype(np.float64).transpose(1, 0)[0]
    # for find peak value


def match_test_2():
    top, bottom, left, right = 100, 100, 0, 500
    img2 = cv2.imread(r'F:\1_sheng\image_stitch\img2jpg\10_max/1850,300000.jpg')
    img1 = cv2.imread(r'F:\1_sheng\image_stitch\img2jpg\10_max/1760,350000.jpg')
    # cv2.copyMakeBorder功能扩充src的边缘，将图像变大，然后以各种外插方式自动填充图像边界,
    # 这个函数实际上调用了函数cv::borderInterpolate，这个函数最重要的功能就是为了处理边界，
    # 比如均值滤波或者中值滤波中，使用copyMakeBorder将原图稍微放大，然后我们就可以处理边界的情况了
    # top, bottom, left, right分别表示在原图四周扩充边缘的大小
    # borderType：扩充边缘的类型，就是外插的类型，OpenCV中给出以下几种方式
    # *BORDER_REPLICATE(复制法，也就是复制最边缘像素。) aaaaaa|abcdefgh|hhhhhhh
    # *BORDER_REFLECT   (轴对称法，也就是以边界为轴，对称。)fedcba|abcdefgh|hgfedcb
    # *BORDER_REFLECT_101(轴对称法，也就是以最边缘像素为轴，对称。)hgfedcb|abcdefgh|gfedcba
    # *BORDER_WRAP    cdefgh|abcdefgh|abcdefg
    # *BORDER_CONSTANT(常量法)  iiiiii|abcdefgh|iiiiiii  with some specified 'i'
    srcImg = cv2.copyMakeBorder(img1, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    testImg = cv2.copyMakeBorder(img2, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    # BGR转换到灰度空间（opencv默认的彩色图像的颜色空间是BGR）
    img1gray = cv2.cvtColor(srcImg, cv2.COLOR_BGR2GRAY)
    img2gray = cv2.cvtColor(testImg, cv2.COLOR_BGR2GRAY)
    # 得到特征提取器的一个实例
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    # 计算出图像的关键点和sift特征向量，img1gray表示输入的原始图
    # 像，可以使三通道或单通道图像
    kp1, des1 = sift.detectAndCompute(img1gray, None)
    kp2, des2 = sift.detectAndCompute(img2gray, None)
    # FLANN parameters随机kd树，平行搜索。默认trees=4
    FLANN_INDEX_KDTREE = 1
    # 创建字典
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    print(index_params)
    # 指定递归遍历的次数checks
    search_params = dict(checks=50)
    print(search_params)
    # 最近邻近似匹配,所以当我们需要找到一个相对好的匹配但是不需要最佳匹配的时候往往使用FlannBasedMatcher。
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    # des1：图片，des2：搜索的图片，matches：匹配的结果，K：阈值，越高精度越高，匹配的数量越少。
    # 该函数，一组返回的俩个DMatch类型DMatch。那么这个这个DMatch数据结构究竟是什么呢？
    # 它包含三个非常重要的数据分别是queryIdx，trainIdx，distance。
    # 先说一下这三个分别是什么在演示其用途：
    # queryIdx：测试图像的特征点描述符的下标（第几个特征点描述符），同时也是描述符对应特征点的下标。
    # trainIdx：样本图像的特征点描述符下标,同时也是描述符对应特征点的下标。
    # distance：代表这怡翠匹配的特征点描述符的欧式距离，数值越小也就说明俩个特征点越相近。
    # K近邻匹配，在匹配的时候选择K个和特征点最相似的点，如果这K个点之间的区别足够大，
    # 则选择最相似的那个点作为匹配点，通常选择K = 2，也就是最近邻匹配。
    # 对每个匹配返回两个最近邻的匹配，如果第一匹配和第二匹配距离比率足够大（向量距离足够远），
    # 则认为这是一个正确的匹配，比率的阈值通常在2左右。
    matches = flann.knnMatch(des1, des2, k=2)
    print('len(matches):', len(matches))
    # for i, matche in enumerate(matches):
    #     print(i, matche)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]
    good = []
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
            matchesMask[i] = [1, 0]
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)
    img3 = cv2.drawMatchesKnn(img1gray, kp1, img2gray, kp2, matches, None, **draw_params)
    plt.imshow(img3, ), plt.show()
    rows, cols = srcImg.shape[:2]
    print('len(good):', len(good))
    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        warpImg = cv2.warpPerspective(testImg, np.array(M), (testImg.shape[1], testImg.shape[0]),
                                     flags=cv2.WARP_INVERSE_MAP)
        for col in range(0, cols):
            if srcImg[:, col].any() and warpImg[:, col].any():
                left = col
                break
        for col in range(cols - 1, 0, -1):
            if srcImg[:, col].any() and warpImg[:, col].any():
                right = col
                break
        res = np.zeros([rows, cols, 3], np.uint8)
        for row in range(0, rows):
            for col in range(0, cols):
                if not srcImg[row, col].any():
                    res[row, col] = warpImg[row, col]
                elif not warpImg[row, col].any():
                    res[row, col] = srcImg[row, col]
                else:
                    srcImgLen = float(abs(col - left))
                    testImgLen = float(abs(col - right))
                    alpha = srcImgLen / (srcImgLen + testImgLen)
                    res[row, col] = np.clip(srcImg[row, col] * (1 - alpha) + warpImg[row, col] * alpha, 0, 255)
        # opencv is bgr, matplotlib is rgb
        res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        # show the result
        plt.figure()
        plt.imshow(res)
        plt.show()
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None


def cv_imread(file_path):
    # 读取中文路径下的图片
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    if cv_img.shape[-1] == 4:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGBA2RGB)
    return cv_img


def rotate_img():
    folder = r'F:\1_sheng\image_stitch\img2jpg\5_liang'
    for i, name in enumerate(sorted(os.listdir(folder), reverse=True)):
        img_path = os.path.join(folder, name)
        print(img_path)
        img = cv_imread(img_path)
        img = np.rot90(img, k=2)
        # cv2.imshow("111", img)
        # cv2.waitKey(0)
        cv2.imwrite(img_path, img)


def ORB_Feature(img1, img2):
    # 初始化ORB
    orb = cv2.ORB_create()

    # 寻找关键点
    kp1 = orb.detect(img1)
    kp2 = orb.detect(img2)

    # 计算描述符
    kp1, des1 = orb.compute(img1, kp1)
    kp2, des2 = orb.compute(img2, kp2)

    # 画出关键点
    outimg1 = cv2.drawKeypoints(img1, keypoints=kp1, outImage=None)
    outimg2 = cv2.drawKeypoints(img2, keypoints=kp2, outImage=None)

    # 显示关键点
    # import numpy as np
    # outimg3 = np.hstack([outimg1, outimg2])
    # cv2.imshow("Key Points", outimg3)
    # cv2.waitKey(0)

    # 初始化 BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # 对描述子进行匹配
    matches = bf.match(des1, des2)

    # 计算最大距离和最小距离
    min_distance = matches[0].distance
    max_distance = matches[0].distance
    for x in matches:
        if x.distance < min_distance:
            min_distance = x.distance
        if x.distance > max_distance:
            max_distance = x.distance

    # 筛选匹配点
    '''
        当描述子之间的距离大于两倍的最小距离时，认为匹配有误。
        但有时候最小距离会非常小，所以设置一个经验值30作为下限。
    '''
    good_match = []
    for x in matches:
        if x.distance <= max(2 * min_distance, 30):
            good_match.append(x)

    # 绘制匹配结果
    draw_match(img1, img2, kp1, kp2, good_match)


def draw_match(img1, img2, kp1, kp2, match):
    cv2.namedWindow("Match Result", 0)
    # cv2.re
    outimage = cv2.drawMatches(img1, kp1, img2, kp2, match, outImg=None)
    cv2.imshow("Match Result", outimage)
    cv2.waitKey(0)


if __name__ == '__main__':
    # cut_video()
    # imageio_cut_video()
    # video2jpg()
    # two_diff()
    # test_fft_2()
    # match_test_2()
    # rotate_img()
    image1 = cv2.imread(r'F:\1_sheng\image_stitch\img2jpg\10\0084.jpg')
    image2 = cv2.imread(r'F:\1_sheng\image_stitch\img2jpg\10\0126.jpg')
    ORB_Feature(image1, image2)

    print("~_~")
