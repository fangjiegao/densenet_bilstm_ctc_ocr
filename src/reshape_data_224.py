# coding=utf8
"""
    image processing
    illool@163.com
"""
import cv2 as cv
import os
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor


# out_dir_ = r"/Users/sherry/data/Synthetic_Chinese_String_Dataset_224"
out_dir_ = r"/home/gaofangjie/Synthetic_Chinese_String_Dataset_224"
# data_dir_ = r"/Users/sherry/data/Synthetic_Chinese_String_Dataset_part"
data_dir_ = r"/home/gaofangjie/Synthetic_Chinese_String_Dataset"

resize_map = {0: cv.INTER_NEAREST,
              1: cv.INTER_LINEAR,
              2: cv.INTER_AREA,
              3: cv.INTER_CUBIC,
              4: cv.INTER_LANCZOS4}


def clamp(pv):
    """防止溢出"""
    if pv > 255:
        return 255
    elif pv < 0:
        return 0
    else:
        return pv


def gaussian_noise_demo(image):
    """添加高斯噪声"""
    h, w, c = image.shape
    for row in range(0, h):
        for col in range(0, w):
            s = np.random.normal(0, 15, 3)  # 产生随机数，每次产生三个
            b = image[row, col, 0]  # blue
            g = image[row, col, 1]  # green
            r = image[row, col, 2]  # red
            image[row, col, 0] = clamp(b + s[0])
            image[row, col, 1] = clamp(g + s[1])
            image[row, col, 2] = clamp(r + s[2])
    return image


def reshape_picture_224_main(data_dir, out_dir):
    im_fns = os.listdir(os.path.join(data_dir, "image"))  # 图片目录
    # 创建目录
    if os.path.exists(os.path.join(out_dir)):
        print(os.path.join(out_dir))
        if os.path.exists(os.path.join(out_dir, "image")):
            print(os.path.join(out_dir, "image"))
        else:
            os.mkdir(os.path.join(out_dir, "image"))
            print("mkdir:", os.path.join(out_dir, "image"))
    else:
        os.mkdir(os.path.join(out_dir))
        print("mkdir:", os.path.join(out_dir))
        os.mkdir(os.path.join(out_dir, "image"))
        print("mkdir:", os.path.join(out_dir, "image"))

    if os.path.exists(os.path.join(out_dir, "image")):
        print(os.path.join(out_dir, "image"))
    else:
        os.mkdir(os.path.join(out_dir, "image"))
        print("mkdir:", os.path.join(out_dir, "image"))
    # 图片转换
    lst = []
    count = 0
    th = ThreadPoolExecutor(os.cpu_count() * 4)
    for im_fn in tqdm(im_fns):
        guss = np.random.randint(0, 5)
        if guss == 0:
            if im_fn.endswith(".jpg"):
                img_path = os.path.join(data_dir, "image", im_fn)
                print(img_path)
                t_r = th.submit(reshape_picture_224_write, img_path, out_dir, im_fn)
                lst.append(t_r)
            count += 1
    th.shutdown()
    print(count, "done......")


def reshape_picture_224_write(img_path, out_dir, im_fn):
    img = cv.imread(img_path)  # img is numpy.array
    if img is not None:
        img = gaussian_noise_demo(img)  # 添加噪声
        x, y = img.shape[0:2]
        r_x = 16. / x
        img_16 = cv.resize(img, (0, 0), fx=r_x, fy=r_x,
                           interpolation=resize_map[np.random.randint(0, 5)])
        r_x = 224. / 16.
        img_224 = cv.resize(img_16, (0, 0), fx=r_x, fy=r_x,
                            interpolation=resize_map[np.random.randint(0, 5)])
        cv.imwrite(os.path.join(out_dir, "image", im_fn), img_224)
        print(os.path.join(out_dir, "image", im_fn))


def reshape_picture_224(img_path):
    img = cv.imread(img_path)  # img is numpy.array
    x, y = img.shape[0:2]
    r_x = 16. / x
    img_16 = cv.resize(img, (0, 0), fx=r_x, fy=r_x, interpolation=cv.INTER_NEAREST)
    r_x = 224. / 16.
    img_224 = cv.resize(img_16, (0, 0), fx=r_x, fy=r_x, interpolation=cv.INTER_NEAREST)
    return img_224, img_224.shape[1]


if __name__ == '__main__':
    reshape_picture_224_main(data_dir_, out_dir_)
    print("nohup python3 reshape_data_224.py > /home/gaofangjie/pro_data.txt 2>&1&")
    print("tail -f pro_data.txt")
    '''
    img_, img_w_ = reshape_picture_224(
        "/Users/sherry/data/Synthetic_Chinese_String_Dataset_part/image/20444625_1057622119.jpg")
    print(type(img_), type(img_w_))
    cv.imshow(str(img_w_), img_)
    cv.waitKey()
    '''
