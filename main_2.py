import numpy as np
import cv2


def cvshow(name, img):
    cv2.imshow(name, img)


def sift_kp(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # sift = cv2.SIFT_create()
    sift = cv2.AKAZE_create()
    kp, des = sift.detectAndCompute(image, None)
    kp_image = cv2.drawKeypoints(gray_image, kp, None)
    return kp_image, kp, des


def get_good_match(des1, des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)  # des1为模板图，des2为匹配图
    matches = sorted(matches, key=lambda x: x[0].distance / x[1].distance)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good


def drawMatches(imageA, imageB, kpsA, kpsB, matches, status):
    # 初始化可视化图片，将A、B图左右连接到一起
    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = imageA
    vis[0:hB, wA:] = imageB

    # 联合遍历，画出匹配对
    for ((trainIdx, queryIdx), s) in zip(matches, status):
        # 当点对匹配成功时，画到可视化图上
        if s == 1:
            # 画出匹配对
            ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
            ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
            cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

    # 返回可视化结果
    return vis


# 全景拼接
def siftimg_rightlignment(img_right, img_left):
    _, kp1, des1 = sift_kp(img_right)
    _, kp2, des2 = sift_kp(img_left)
    goodMatch = get_good_match(des1, des2)
    # goodMatch = goodMatch[:5]
    # 当筛选项的匹配对大于4对时：计算视角变换矩阵
    if len(goodMatch) > 4:
        # 获取匹配对的点坐标
        ptsA = np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        ransacReprojThreshold = 50
        print(f"ptsA:{ptsA},\nptsB:{ptsB}")
        exit(0)
        a = cv2.estimateAffine2D(ptsA, ptsB)
        x_offset = int(a[0][0][2]) + 10
        y_offset = int(a[0][1][2])
        print("---------: ", a[0], x_offset, y_offset)
        print("img_left.shape:", img_left.shape)
        # 拿到x方向的平移参数，直接把下一张图片的后面那个平移参数数量的像素列加到上一张图片后面就好
        new_img = np.zeros((img_left.shape[0], img_left.shape[1]+x_offset+1, 3))
        print(new_img.shape)
        # print(new_img)
        new_img[0:img_left.shape[0], 0:img_left.shape[1]] = img_left

        temp_next_crop = img_right[:, img_right.shape[1]-x_offset:]
        cv2.imwrite("./1_2_temp_next_crop.jpg", temp_next_crop)
        # print(new_img)
        crop_concat_img = np.concatenate((img_left, temp_next_crop), axis=1)
        crop_concat_img = rotate_bound(crop_concat_img, 90)
        cv2.imshow("crop_concat_img", crop_concat_img)
        cv2.imshow("temp_next_crop", temp_next_crop)
        cv2.waitKey(0)

        # cv2.warpAffine()
        H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold)
        #  该函数的作用就是先用RANSAC选择最优的四组配对点，再计算H矩阵。H为3*3矩阵

        # 拿右下角的点进行透视变化，用于固定输出图片的宽，
        # print("org right shape:", img_right.shape)
        temp_box = np.array([[[0, 0], [img_right.shape[1], img_right.shape[0]]]], dtype='float32')
        # print("before box: ", temp_box)
        # transform_box = cv2.perspectiveTransform(temp_box, H)
        # print(temp_box, transform_box)
        # 将图片右进行视角变换，result是变换后图片
        print("HHH:", H)
        # result = cv2.warpPerspective(img_right, H, (img_right.shape[1] + img_left.shape[1], img_right.shape[0]))
        result = cv2.warpPerspective(img_right, H, (img_right.shape[1]+img_left.shape[1], img_right.shape[0]))
        cvshow('result_medium', result)
        # print("result.shape:", result.shape)
        # cv2.imshow("new result", result)
        # 将图片左传入result图片最左端
        result[0:img_left.shape[0], 0:img_left.shape[1]] = img_left
        # print("img_left.shape:", img_left.shape)
        # cv2.imshow("new img_left", img_left)
        cv2.waitKey(2)
        return result


def cv_imread(file_path):
    # 读取中文路径下的图片
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    if cv_img.shape[-1] == 4:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGBA2RGB)
    return cv_img


def rotate_bound(image, angle):
    # 获取图像的尺寸
    # 旋转中心
    (h, w) = image.shape[:2]
    (cx, cy) = (w / 2, h / 2)

    # 设置旋转矩阵
    M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # 计算图像旋转后的新边界
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # 调整旋转矩阵的移动距离（t_{x}, t_{y}）
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    return cv2.warpAffine(image, M, (nW, nH))


if __name__ == '__main__':
    # 特征匹配＋全景拼接

    # 读取拼接图片（注意图片左右的放置）
    # 是对右边的图形做变换
    # img_right = cv2.imread(r'd:/right1.jpg')
    # img_left = cv2.imread(r'd:/left1.jpg')
    img_1 = cv_imread(r"./抓图图示/0位置/5位置.jpg")
    img_2 = cv_imread(r"./抓图图示/25位置/25位置.jpg")
    img_3 = cv_imread(r"./抓图图示/50位置/55位置.jpg")
    img_4 = cv_imread(r"./抓图图示/75位置/70位置.jpg")
    img_5 = cv_imread(r"./抓图图示/100位置/100位置（最低）.jpg")

    # img_1 = cv2.GaussianBlur(img_1, (5, 5), 0)
    # img_2 = cv2.GaussianBlur(img_2, (5, 5), 0)
    # img_left = cv2.bilateralFilter(img_2, 9, 41, 41)
    # img_right = cv2.bilateralFilter(img_3, 9, 41, 41)
    img_left = rotate_bound(img_1, 270)
    img_right = rotate_bound(img_2, 270)

    img_right = cv2.resize(img_right, None, fx=0.5, fy=0.5)
    # 保证两张图一样大
    img_left = cv2.resize(img_left, (img_right.shape[1], img_right.shape[0]))

    cv2.imshow("img_left", img_left)
    cv2.imshow("img_right", img_right)
    cv2.waitKey(4)
    cv2.destroyAllWindows()

    kpimg_right, kp1, des1 = sift_kp(img_right)
    kpimg_left, kp2, des2 = sift_kp(img_left)

    # 同时显示原图和关键点检测后的图
    # cvshow('img_left', np.hstack((img_left, kpimg_left)))
    # cvshow('img_right', np.hstack((img_right, kpimg_right)))
    goodMatch = get_good_match(des1, des2)

    all_goodmatch_img = cv2.drawMatches(img_right, kp1, img_left, kp2, goodMatch, None, flags=2)

    # goodmatch_img自己设置前多少个goodMatch[:10]
    goodmatch_img = cv2.drawMatches(img_right, kp1, img_left, kp2, goodMatch[:10], None, flags=2)

    cvshow('Keypoint Matches1', all_goodmatch_img)
    goodmatch_img = rotate_bound(goodmatch_img, 90)
    cvshow('Keypoint Matches2', goodmatch_img)

    # 把图片拼接成全景图
    result = siftimg_rightlignment(img_right, img_left)
    result = rotate_bound(result, 90)
    # 裁掉下面全黑的
    print("all result.shape:", result.shape)
    # for i in range(result.shape[0], 0, -1):
    #     if sum(sum(sum(result[i-1:i]))) != 0:
    #         result = result[0:i, :]
    #         break
    print("all result.shape:", result.shape)
    cvshow('result', result)
    cv2.imwrite("./1_2result.jpg", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
