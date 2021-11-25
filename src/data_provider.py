# encoding:utf-8
import os
import time
import numpy as np
import re
import data_util
import cv2 as cv

# 数据集目录
# DATA_FOLDER = "data/dataset/mlt/"
IMG_FOLDER = "/home/gaofangjie/Synthetic_Chinese_String_Dataset"
LABEL_FOLDER = "/home/gaofangjie/Synthetic_Chinese_String_Dataset_label"

lable_txt = None
re_str = r'$file_name.jpg (.+)\n'
lable_dic = None
lable_dic_opposite = None

resize_map = {0: cv.INTER_NEAREST,
              1: cv.INTER_LINEAR,
              2: cv.INTER_AREA,
              3: cv.INTER_CUBIC,
              4: cv.INTER_LANCZOS4}


def get_training_data(img_dir):
    img_files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    # for parent, dirnames, filenames in os.walk(os.path.join(DATA_FOLDER, "image")):
    print(os.path.join(img_dir, "image"))
    for parent, dirnames, filenames in os.walk(os.path.join(img_dir, "image")):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    img_files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(img_files)))
    return img_files


def load_annoataion(p):
    bbox = []
    with open(p, "r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split(",")
        x_min, y_min, x_max, y_max = map(int, line)
        bbox.append([x_min, y_min, x_max, y_max, 1])
    return bbox


def load_lable(lable_path, file_name):
    global lable_txt
    global lable_dic
    if lable_txt is None:
        label_file = os.path.join(lable_path, "label", "360_train.txt")

        with open(label_file, "r", encoding='utf-8') as f:
            lable_txt = f.read()
    if lable_dic is None:
        lable_dic = read_lable_dict(lable_path)

    lable_group = re.compile(re_str.replace("$file_name", file_name)).search(lable_txt)
    if lable_group is not None:
        lable = lable_group.group(1)
        lable = lable.replace(" ", "")
    else:
        lable = ""

    if lable != "":
        lable_idx = [int(lable_dic[_]) for _ in list(lable)]
    else:
        lable_idx = []
    return lable_idx, lable


def read_lable_dict(lable_path):
    with open(os.path.join(lable_path, "lable_set.txt"), 'r', encoding='utf-8') as f:
        text = f.read()
    lable_list = text.split("\n")
    lable_index = list(range(len(lable_list)))
    lable_set = dict(zip(lable_list, lable_index))
    return lable_set


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


def reshape_picture_224_for_th(img_path):
    img = cv.imread(img_path)   # img is numpy.array
    if img is not None:
        is_gaussian_noise = np.random.randint(0, 5)
        if is_gaussian_noise == 0:
            img = gaussian_noise_demo(img)  # 添加噪声
        x, y = img.shape[0:2]
        r_x = 16. / x
        img_16 = cv.resize(img, (0, 0), fx=r_x, fy=r_x,
                           interpolation=resize_map[np.random.randint(0, 5)])
        r_x = 224. / 16.
        img_224 = cv.resize(img_16, (0, 0), fx=r_x, fy=r_x,
                            interpolation=resize_map[np.random.randint(0, 5)])
        return img_224, img_224.shape[1], img_224.shape
    return None, None, None


def generator(img_dir, label_dir, vis=False):
    print("generator..............")
    image_list = np.array(get_training_data(img_dir))
    print('{} training images in {}'.format(image_list.shape[0], img_dir))
    index = np.arange(0, image_list.shape[0])
    while True:
        np.random.shuffle(index)
        for i in index:
            try:
                filepath, fullflname = os.path.split(image_list[i])
                fname, ext = os.path.splitext(fullflname)
                img_s, img_w, img_shape = reshape_picture_224_for_th(image_list[i])
                img = img_s * (1. / 255)
                img = np.expand_dims(img, axis=0)
                img_w = np.expand_dims(img_w, axis=0)
                lable_idx, lable = load_lable(label_dir, fname)
                lable_seq_len = np.expand_dims(len(lable_idx), axis=0)
                lable_idx = np.expand_dims(lable_idx, axis=0)
                img_shape = np.expand_dims(img_shape, axis=0)

                if vis:
                    # cv.imshow("img", img[0])
                    print(img.shape)
                    print(img_w)
                    print(img_shape)
                    print(lable_idx)
                    print(lable)
                    print("lable_seq_len:", lable_seq_len)
                    # cv2.waitKey()
                yield img, img_w, img_shape, lable_idx, lable_seq_len

            except Exception as e:
                print(e)
                continue


def get_batch(num_workers, **kwargs):
    enqueuer = None
    try:
        # enqueuer = data_util.GeneratorEnqueuer(generator(**kwargs), use_multiprocessing=True)
        enqueuer = data_util.GeneratorEnqueuer(generator(**kwargs), use_multiprocessing=False)
        enqueuer.start(max_queue_size=5, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()


if __name__ == '__main__':
    gen = get_batch(num_workers=2, img_dir=IMG_FOLDER, label_dir=LABEL_FOLDER, vis=True)
    while True:
        img_, img_w_, img_shape_, lable_idx_, lable_seq_len_ = next(gen)
        print('done')
