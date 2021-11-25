# coding=utf-8

"""
Generator tfrecord file tool class
illool@163.com
"""

import os
import tensorflow as tf
import cv2 as cv
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import re
import gen_and_get_lable_set
import random


lable_txt = None
re_str = r'$file_name.jpg (.+)\n'
lable_dic = None
lable_dic_opposite = None


class GeneratorTfrecordTool(object):
    @staticmethod
    def convert_to_example_simple(image_example, image_buffer):
        """
        covert to tf.train.Example
        :param image_example: dict, an image example
        :param image_buffer: numpy.array, JPEG encoding of RGB image
        :return: Example proto
        """
        image_info = image_example['shape']  # [[816 608   3]]
        bboxs = np.array(image_example['bboxs'])  # 传入bbox [[x_min, y_min, x_max, y_max, 1],[]]
        bboxs_shape = list(bboxs.shape)  # [21, 5],21个小box,[x_min, y_min, x_max, y_max, 1]1:包含目标;0:不包含目标
        image_encoded = image_buffer.tostring() if isinstance(image_buffer, np.ndarray) else image_buffer
        # example
        example = tf.train.Example(features=tf.train.Features(feature={
            'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_encoded])),  # image
            'image/image_info': tf.train.Feature(int64_list=tf.train.Int64List(value=image_info[0])),  # image_shape
            'image/bboxs_info': tf.train.Feature(int64_list=tf.train.Int64List(value=bboxs_shape)),  # bboxs shape
            'image/bboxs': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bboxs.tostring()]))  # landmark坐标
        }))
        return example

    @staticmethod
    def convert_to_example_simple_synthetic_chinese_string_dataset(lable_example, image_buffer):
        """
        covert to tf.train.Example
        :param lable_example: dict, an image example
        :param image_buffer: numpy.array, JPEG encoding of RGB image
        :return: Example proto
        """
        image_info = lable_example['shape']  # [[816 608   3]]
        image_w = image_info[0][1]
        lable_seq = lable_example['lable_seq']  # 传入lable index[1,2,3,4,5]
        lable_seq_len = len(lable_seq)
        lable_seq = np.array(lable_seq)  # for tostring()
        image_encoded = image_buffer.tostring() if isinstance(image_buffer, np.ndarray) else image_buffer
        # example
        example = tf.train.Example(features=tf.train.Features(feature={
            'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_encoded])),  # image
            'image/image_info': tf.train.Feature(int64_list=tf.train.Int64List(value=image_info[0])),  # image_shape
            'image/lable_seq_len': tf.train.Feature(int64_list=tf.train.Int64List(value=[lable_seq_len])),  # len
            'image/image_w': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_w])),  # image_w
            'image/lable_seq': tf.train.Feature(bytes_list=tf.train.BytesList(value=[lable_seq.tostring()]))  # lable
        }))
        return example

    @staticmethod
    def read_img_and_bboxs(img_path, bbox_path):
        img = cv.imread(img_path)  # img is numpy.array  [[2 3 4]]
        h, w, c = img.shape
        im_info = np.array([h, w, c]).reshape([1, 3])  # [[h, w, c]]
        bboxs = GeneratorTfrecordTool.load_annoataion(bbox_path)
        image_example = {"shape": im_info, "bboxs": bboxs}
        return image_example, img

    @staticmethod
    def read_img_and_bboxs_gfile(img_path, bbox_path):
        img = cv.imread(img_path)  # img is numpy.array
        h, w, c = img.shape
        im_info = np.array([h, w, c]).reshape([1, 3])  # [[h, w, c]]
        bboxs = GeneratorTfrecordTool.load_annoataion(bbox_path)
        image_example = {"shape": im_info, "bboxs": bboxs}
        # img = tf.gfile.FastGFile(img_path, 'rb').read()  # img is bytes
        g_img = tf.gfile.GFile(img_path, 'rb').read()  # img is bytes
        return image_example, g_img

    @staticmethod
    def read_img_and_lable_gfile_synthetic_chinese_string_dataset(img_path, lable_path, file_name):
        img = cv.imread(img_path)  # img is numpy.array
        h, w, c = img.shape
        im_info = np.array([h, w, c]).reshape([1, 3])
        lable_idx = GeneratorTfrecordTool.load_lable_synthetic_chinese_string_dataset(lable_path, file_name)
        print("lable_idx:", lable_idx)
        lable_example = {"shape": im_info, "lable_seq": lable_idx}
        # img = tf.gfile.FastGFile(img_path, 'rb').read()  # img is bytes
        g_img = tf.gfile.GFile(img_path, 'rb').read()  # img is bytes
        return lable_example, g_img

    @staticmethod
    def load_annoataion(bbox_path):
        bboxs = []
        with open(bbox_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().split(",")
            x_min, y_min, x_max, y_max = map(int, line)
            bboxs.append([x_min, y_min, x_max, y_max, 1])
        return bboxs

    @staticmethod
    def load_lable_synthetic_chinese_string_dataset(lable_path, file_name):
        global lable_txt
        global lable_dic
        if lable_txt is None:
            with open(lable_path, "r") as f:
                lable_txt = f.read()
        if lable_dic is None:
            lable_dic = gen_and_get_lable_set.read_lable_dict()

        lable_group = re.compile(re_str.replace("$file_name", file_name)).search(lable_txt)
        if lable_group is not None:
            lable = lable_group.group(1)
        else:
            lable = ""

        if lable != "":
            lable_idx = [int(lable_dic[_]) for _ in list(lable)]
        else:
            lable_idx = []
        return lable_idx

    @staticmethod
    def convert_to_tfrecord_data(data_dir, out_path):
        """
        covert to tfrecord file
        :param data_dir: string, trian dir
        :param out_path: string, tfrecord file dir
        :return:
        """
        tf_writer = tf.python_io.TFRecordWriter(out_path)
        im_fns = os.listdir(os.path.join(data_dir, "image"))  # 图片目录
        for im_fn in tqdm(im_fns):
            # noinspection PyBroadException
            try:
                _, fn = os.path.split(im_fn)
                bfn, ext = os.path.splitext(fn)  # 将文件名和扩展名分开
                if ext.lower() not in ['.jpg', '.png']:
                    continue

                bboxs_path = os.path.join(data_dir, "label", '' + bfn + '.txt')
                img_path = os.path.join(data_dir, "image", im_fn)
                # 原始文件的方式转换数据
                image_example, img = GeneratorTfrecordTool.read_img_and_bboxs(img_path, bboxs_path)
                # image_example, img = GeneratorTfrecordTool.read_img_and_bboxs_gfile(img_path, bboxs_path)
                example = GeneratorTfrecordTool.convert_to_example_simple(image_example, img)
                tf_writer.write(example.SerializeToString())
            except Exception as e:
                print(e)
                print("Error processing {}".format(im_fn))
        tf_writer.close()

    @staticmethod
    def convert_to_tfrecord_data_gfile(data_dir, out_path):
        """
        covert to tfrecord file
        :param data_dir: string, trian dir
        :param out_path: string, tfrecord file dir
        :return:
        """
        tf_writer = tf.python_io.TFRecordWriter(out_path)
        im_fns = os.listdir(os.path.join(data_dir, "image"))  # 图片目录
        for im_fn in tqdm(im_fns):
            # noinspection PyBroadException
            try:
                _, fn = os.path.split(im_fn)
                bfn, ext = os.path.splitext(fn)  # 将文件名和扩展名分开
                if ext.lower() not in ['.jpg', '.png']:
                    continue

                bboxs_path = os.path.join(data_dir, "label", '' + bfn + '.txt')
                img_path = os.path.join(data_dir, "image", im_fn)
                # image_example, img = GeneratorTfrecordTool.read_img_and_bboxs(img_path, bboxs_path)
                # gfile的方式转换数据
                image_example, img = GeneratorTfrecordTool.read_img_and_bboxs_gfile(img_path, bboxs_path)
                example = GeneratorTfrecordTool.convert_to_example_simple(image_example, img)
                tf_writer.write(example.SerializeToString())
            except Exception as e:
                print(e)
                print("Error processing {}".format(im_fn))
        tf_writer.close()

    @staticmethod
    def convert_to_tfrecord_data_slice_gfile_synthetic_chinese_string_dataset(data_dir, lable_dir, out_dir, tf_name):
        """
        covert to tfrecord file
        :param data_dir: string, trian dir
        :param lable_dir: string, lable dir
        :param out_dir: string, tfrecord file dir
        :param tf_name: string, tfrecord file name
        :return:
        """
        im_fns = os.listdir(os.path.join(data_dir, "image"))  # 图片目录
        writer_num = 0
        im_num = 0
        tf_writer = None
        for im_fn in tqdm(im_fns):
            # noinspection PyBroadException
            if im_num == 0:
                if tf_writer is not None:
                    tf_writer.close()
                tf_writer = tf.python_io.TFRecordWriter(out_dir + os.sep + tf_name.replace("x", str(writer_num)))
                writer_num += 1
                print(writer_num, out_dir + os.sep + tf_name.replace("x", str(writer_num)))
                pass

            try:
                _, fn = os.path.split(im_fn)
                bfn, ext = os.path.splitext(fn)  # 将文件名和扩展名分开
                if ext.lower() not in ['.jpg', '.png']:
                    continue

                bboxs_path = os.path.join(lable_dir, "label", '' + '360_train' + '.txt')
                img_path = os.path.join(data_dir, "image", im_fn)
                # image_example, img = GeneratorTfrecordTool.read_img_and_bboxs(img_path, bboxs_path)
                # gfile的方式转换数据
                lable_example, img = GeneratorTfrecordTool.read_img_and_lable_gfile_synthetic_chinese_string_dataset(
                    img_path, bboxs_path, bfn)
                example = GeneratorTfrecordTool.convert_to_example_simple_synthetic_chinese_string_dataset(
                    lable_example, img)
                tf_writer.write(example.SerializeToString())
                im_num += 1
                if im_num >= 1:  # 10000
                    im_num = 0
            except Exception as e:
                print(e)
                print("Error processing {}".format(im_fn))
        if tf_writer is not None:
            tf_writer.close()

    @staticmethod
    def convert_to_tfrecord_data_gfile_synthetic_chinese_string_dataset(data_dir, lable_dir, out_path):
        """
        covert to tfrecord file
        :param data_dir: string, trian dir
        :param lable_dir: string, lable dir
        :param out_path: string, tfrecord file dir
        :return:
        """
        tf_writer = tf.python_io.TFRecordWriter(out_path)
        im_fns = os.listdir(os.path.join(data_dir, "image"))  # 图片目录
        for im_fn in tqdm(im_fns):
            # noinspection PyBroadException
            try:
                _, fn = os.path.split(im_fn)
                bfn, ext = os.path.splitext(fn)  # 将文件名和扩展名分开
                if ext.lower() not in ['.jpg', '.png']:
                    continue

                bboxs_path = os.path.join(lable_dir, "label", '' + '360_train' + '.txt')
                img_path = os.path.join(data_dir, "image", im_fn)
                # image_example, img = GeneratorTfrecordTool.read_img_and_bboxs(img_path, bboxs_path)
                # gfile的方式转换数据
                lable_example, img = GeneratorTfrecordTool.read_img_and_lable_gfile_synthetic_chinese_string_dataset(
                    img_path, bboxs_path, bfn)
                example = GeneratorTfrecordTool.convert_to_example_simple_synthetic_chinese_string_dataset(
                    lable_example, img)
                tf_writer.write(example.SerializeToString())
            except Exception as e:
                print(e)
                print("Error processing {}".format(im_fn))
        tf_writer.close()

    @staticmethod
    def read_and_decode_tfrecord_data_and_show(tfrecord_name):

        def _parse_record(example_proto):
            features_ = {
                'image/image_info': tf.FixedLenFeature([3], tf.int64),  # tf.int64,必须指定长度:3
                'image/bboxs_info': tf.FixedLenFeature([2], tf.int64),  # tf.int64,必须指定长度:2
                'image/encoded': tf.FixedLenFeature([], tf.string),
                'image/bboxs': tf.FixedLenFeature([], tf.string)
            }
            parsed_features = tf.parse_single_example(example_proto, features=features_)
            return parsed_features

        # 用 dataset 读取 tfrecord 文件
        dataset = tf.data.TFRecordDataset(tfrecord_name)
        dataset = dataset.map(_parse_record)
        iterator = dataset.make_one_shot_iterator()
        num = GeneratorTfrecordTool.total_sample(tfrecord_name)
        with tf.Session() as sess:
            for i in range(num):
                features = sess.run(iterator.get_next())
                img_data = features['image/encoded']
                shape = features['image/image_info']
                bboxs_shape = features['image/bboxs_info']
                bboxs = features['image/bboxs']
                bboxs = tf.decode_raw(bboxs, tf.int64)
                bboxs = tf.reshape(bboxs, bboxs_shape)
                bboxs_list = bboxs.eval()

                img_data = tf.decode_raw(img_data, tf.uint8)
                # img_data = tf.image.decode_jpeg(img_data)
                img_data = tf.reshape(img_data, shape)
                img_data = tf.cast(img_data, tf.float32) / 255.  # 必须除以255.

                plt.figure()
                # 显示图片
                plt.imshow(img_data.eval())
                for _ in bboxs_list:
                    plt.plot([_[0], _[2], _[2], _[0], _[0]], [_[1], _[1], _[3], _[3], _[1]], 'g', '-.')
                plt.show()
                plt.show()

                # 将数据重新编码成 jpg 图片并保存
                # img = tf.image.encode_jpeg(img_data)
                # tf.gfile.GFile('cat_encode.jpg', 'wb').write(img_data.eval())

    @staticmethod
    def read_and_decode_gfile_tfrecord_data_and_show(tfrecord_name):

        def _parse_record(example_proto):
            features_ = {
                'image/image_info': tf.FixedLenFeature([3], tf.int64),  # tf.int64,必须指定长度:3
                'image/bboxs_info': tf.FixedLenFeature([2], tf.int64),  # tf.int64,必须指定长度:2
                'image/encoded': tf.FixedLenFeature([], tf.string),
                'image/bboxs': tf.FixedLenFeature([], tf.string)
            }
            parsed_features = tf.parse_single_example(example_proto, features=features_)
            return parsed_features

        # 用 dataset 读取 tfrecord 文件g
        dataset = tf.data.TFRecordDataset(tfrecord_name)
        dataset = dataset.map(_parse_record)
        iterator = dataset.make_one_shot_iterator()
        num = GeneratorTfrecordTool.total_sample(tfrecord_name)
        with tf.Session() as sess:
            for i in range(num):
                features = sess.run(iterator.get_next())
                img_data = features['image/encoded']
                shape = features['image/image_info']
                bboxs_shape = features['image/bboxs_info']
                bboxs = features['image/bboxs']
                bboxs = tf.decode_raw(bboxs, tf.int64)
                bboxs = tf.reshape(bboxs, bboxs_shape)
                bboxs_list = bboxs.eval()
                # 恢复为.jpg文件
                # tf.gfile.GFile(str(i) + '.jpg', 'wb').write(img_data)

                img_data = tf.image.decode_jpeg(img_data)
                # img_data = tf.decode_raw(img_data, tf.uint8)
                # image_data = np.reshape(img_data, shape)
                img_data = tf.reshape(img_data, shape)
                img_data = tf.cast(img_data, tf.float32) / 255.  # 必须除以255.

                plt.figure()
                # 显示图片
                plt.imshow(img_data.eval())
                for _ in bboxs_list:
                    plt.plot([_[0], _[2], _[2], _[0], _[0]], [_[1], _[1], _[3], _[3], _[1]], 'r', '-.')
                plt.show()

                # 将数据重新编码成 jpg 图片并保存
                # img_data = features['image/encoded']
                # tf.gfile.GFile(str(i) + '.jpg', 'wb').write(img_data)

    # 该函数用于统计 TFRecord 文件中的样本数量(总数)
    @staticmethod
    def total_sample(tfrecord_name):
        sample_nums = 0
        for record in tf.python_io.tf_record_iterator(tfrecord_name):
            type(record)
            sample_nums += 1
        return sample_nums

    @staticmethod
    def total_sample_by_path(tfrecord_path):
        im_fns = os.listdir(tfrecord_path)  # 图片目录
        im_fns = [tfrecord_path + os.sep + _ for _ in im_fns]
        im_nums = [GeneratorTfrecordTool.total_sample(_) for _ in im_fns]
        num = sum(im_nums)
        return num

    @staticmethod
    def read_gfile_tfrecord_data(tfrecord_name):

        features_ = {
            'image/image_info': tf.FixedLenFeature([3], tf.int64),  # tf.int64,必须指定长度:3
            'image/bboxs_info': tf.FixedLenFeature([2], tf.int64),  # tf.int64,必须指定长度:2
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/bboxs': tf.FixedLenFeature([], tf.string)
        }

        # 根据文件名生成一个队列
        filename_queue = tf.train.string_input_producer([tfrecord_name])

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
        features = tf.parse_single_example(serialized_example,
                                           features=features_)
        img_data = features['image/encoded']
        shape = features['image/image_info']
        bboxs_shape = features['image/bboxs_info']
        bboxs = features['image/bboxs']
        bboxs = tf.decode_raw(bboxs, tf.int64)
        print("bboxs_shape:", bboxs_shape)
        bboxs = tf.reshape(bboxs, bboxs_shape)
        # bboxs_list = bboxs.eval()
        img_data = tf.image.decode_jpeg(img_data)
        img_data = tf.reshape(img_data, shape)
        img_data = tf.cast(img_data, tf.float32) * (1. / 255) - 0.5

        return img_data, shape, bboxs, bboxs_shape

    @staticmethod
    def read_gfile_synthetic_chinese_string_dataset_tfrecord_data(tfrecord_name):

        features_ = {
            'image/image_info': tf.FixedLenFeature([3], tf.int64),  # tf.int64,必须指定长度:3
            'image/lable_seq_len': tf.FixedLenFeature([1], tf.int64),  # tf.int64,必须指定长度:1
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/lable_seq': tf.FixedLenFeature([], tf.string),
            'image/image_w': tf.FixedLenFeature([1], tf.int64),  # tf.int64,必须指定长度:1
        }

        # 根据文件名生成一个队列
        filename_queue = tf.train.string_input_producer([tfrecord_name])

        reader = tf.TFRecordReader()
        # reader = tf.WholeFileReader()
        _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
        features = tf.parse_single_example(serialized_example, features=features_)

        img_data = features['image/encoded']
        shape = features['image/image_info']
        lable_seq_len = features['image/lable_seq_len']
        lable_seq = features['image/lable_seq']
        lable_seq = tf.decode_raw(lable_seq, tf.int64)
        # lable_seq = tf.reshape(lable_seq, [1, lable_seq_len])
        lable_seq = tf.expand_dims(lable_seq, 0)
        image_w = features['image/image_w']

        img_data = tf.image.decode_jpeg(img_data)
        img_data = tf.reshape(img_data, shape)
        img_data = tf.expand_dims(img_data, 0)
        img_data = tf.cast(img_data, tf.float32) * (1. / 255) - 0.5

        return img_data, shape, lable_seq, lable_seq_len, image_w

    @staticmethod
    def read_slice_gfile_synthetic_chinese_string_dataset_tfrecord_data(tfrecord_path):

        features_ = {
            'image/image_info': tf.FixedLenFeature([3], tf.int64),  # tf.int64,必须指定长度:3
            'image/lable_seq_len': tf.FixedLenFeature([1], tf.int64),  # tf.int64,必须指定长度:1
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/lable_seq': tf.FixedLenFeature([], tf.string),
            'image/image_w': tf.FixedLenFeature([1], tf.int64),  # tf.int64,必须指定长度:1
        }

        # 根据文件名生成一个队列
        im_fns = os.listdir(tfrecord_path)  # 图片目录
        im_fns = [tfrecord_path + os.sep + _ for _ in im_fns]
        if len(im_fns) > 5:
            im_fns = random.sample(im_fns, 5)  # 因为数据量太大,随机选择5个
        # filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(tfrecord_path))
        filename_queue = tf.train.string_input_producer(im_fns)

        reader = tf.TFRecordReader()
        # reader = tf.WholeFileReader()
        _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
        features = tf.parse_single_example(serialized_example, features=features_)

        img_data = features['image/encoded']
        shape = features['image/image_info']
        lable_seq_len = features['image/lable_seq_len']
        lable_seq = features['image/lable_seq']
        lable_seq = tf.decode_raw(lable_seq, tf.int64)
        # lable_seq = tf.reshape(lable_seq, [1, lable_seq_len])
        lable_seq = tf.expand_dims(lable_seq, 0)
        image_w = features['image/image_w']

        img_data = tf.image.decode_jpeg(img_data)
        img_data = tf.reshape(img_data, shape)
        img_data = tf.expand_dims(img_data, 0)
        img_data = tf.cast(img_data, tf.float32) * (1. / 255) - 0.5

        return img_data, shape, lable_seq, lable_seq_len, image_w

    @staticmethod
    def read_gfile_tfrecord_data_sess(tfrecord_name):
        img_data, shape, bboxs, bboxs_shape = \
            GeneratorTfrecordTool.read_gfile_tfrecord_data(tfrecord_name)

        # 使用shuffle_batch可以随机打乱输入, capacity就是此队列的容量,min_after_dequeue的值一定要比capacity要小
        # 使用shuffle_batch的前提是图片的shape都都一样,本例下是不能使用的
        # img_data, shape, bboxs, bboxs_shape = tf.train.shuffle_batch([img_data, shape, bboxs, bboxs_shape],
        #                                                              batch_size=1, capacity=2000,
        #                                                              min_after_dequeue=1000)
        init = tf.global_variables_initializer()
        num = GeneratorTfrecordTool.total_sample(tfrecord_name)
        coord = tf.train.Coordinator()
        with tf.Session() as sess:
            sess.run(init)
            queue_runner = tf.train.start_queue_runners(sess, coord=coord)
            for i in range(num):
                img_data_, shape_, bboxs_, bboxs_shape_ = sess.run([img_data, shape, bboxs, bboxs_shape])
                # do trian :feed_dict = {img_data_:img_data_....}
                print(img_data_.shape, shape_, bboxs_.shape, bboxs_shape_)
            coord.request_stop()
            coord.join(queue_runner)

    @staticmethod
    def read_gfile_synthetic_chinese_string_dataset_tfrecord_data_sess(tfrecord_name):

        global lable_dic_opposite
        if lable_dic_opposite is None:
            lable_dic_opposite = gen_and_get_lable_set.read_lable_dict_opposite()

        img_data, shape, lable_seq, lable_seq_len, image_w = \
            GeneratorTfrecordTool.read_gfile_synthetic_chinese_string_dataset_tfrecord_data(tfrecord_name)

        lable_seq_sparse = dense2sparse(lable_seq)
        lable_seq_dense = tf.sparse_tensor_to_dense(
            lable_seq_sparse, default_value=-1, validate_indices=True, name=None)
        # 使用shuffle_batch可以随机打乱输入, capacity就是此队列的容量,min_after_dequeue的值一定要比capacity要小
        # 使用shuffle_batch的前提是图片的shape都都一样,本例下是不能使用的
        # img_data, shape, bboxs, bboxs_shape = tf.train.shuffle_batch([img_data, shape, bboxs, bboxs_shape],
        #                                                              batch_size=1, capacity=2000,
        #                                                              min_after_dequeue=1000)

        init = tf.global_variables_initializer()
        num = GeneratorTfrecordTool.total_sample(tfrecord_name)
        coord = tf.train.Coordinator()
        with tf.Session() as sess:
            sess.run(init)
            queue_runner = tf.train.start_queue_runners(sess, coord=coord)
            for i in range(num):
                print("check:", i)
                # 必须放在循环体里面sess.run(数据op)
                img_data_, shape_, lable_seq_, lable_seq_len_, image_w_, lable_seq_dense_ = sess.run(
                    [img_data, shape, lable_seq, lable_seq_len, image_w, lable_seq_dense])

                lable_seq_ = tf.reshape(lable_seq_, [1, lable_seq_len.eval()])
                # do trian :feed_dict = {img_data_:img_data_....}
                print("type img_data:", type(img_data_))
                print("img_data_:", img_data_)
                print(img_data_.shape, shape_, lable_seq_.shape, lable_seq_len_, image_w_)
                print("lable_seq_:", lable_seq_.eval())
                lable_text = [lable_dic_opposite[_] for _ in lable_seq_.eval()[0]]
                print("".join(lable_text))
                print("lable_seq_dense_:", lable_seq_dense_)
                lable_seq_dense_text = [lable_dic_opposite[_] for _ in lable_seq_dense_[0]]
                print("".join(lable_seq_dense_text))
            coord.request_stop()
            coord.join(queue_runner)

            '''
            try:
                while not coord.should_stop():
                    print('************')
                    # 获取每一个batch中batch_size个样本和标签
                    img_data_, shape_, lable_seq_, lable_seq_len_, image_w_ = sess.run(
                        [img_data, shape, lable_seq, lable_seq_len, image_w])

                    lable_seq_ = tf.reshape(lable_seq_, [1, lable_seq_len.eval()])
                    # do trian :feed_dict = {img_data_:img_data_....}
                    print(img_data_.shape, shape_, lable_seq_.shape, lable_seq_len_, image_w_)
                    print("lable_seq_:", lable_seq_.eval())
                    lable_text = [lable_dic_opposite[_] for _ in lable_seq_.eval()[0]]
                    print("".join(lable_text))
            except tf.errors.OutOfRangeError:  # 如果读取到文件队列末尾会抛出此异常
                print("done! now lets kill all the threads……")
            finally:
                # 协调器coord发出所有线程终止信号
                coord.request_stop()
                print('all threads are asked to stop!')
            coord.join(queue_runner)  # 把开启的线程加入主线程，等待threads结束
            print('all threads are stopped!')
            '''

    @staticmethod
    def read_slice_gfile_synthetic_chinese_string_dataset_tfrecord_data_sess(tfrecord_path):

        global lable_dic_opposite
        if lable_dic_opposite is None:
            lable_dic_opposite = gen_and_get_lable_set.read_lable_dict_opposite()

        img_data, shape, lable_seq, lable_seq_len, image_w = \
            GeneratorTfrecordTool.read_slice_gfile_synthetic_chinese_string_dataset_tfrecord_data(tfrecord_path)

        lable_seq_sparse = dense2sparse(lable_seq)
        lable_seq_dense = tf.sparse_tensor_to_dense(
            lable_seq_sparse, default_value=-1, validate_indices=True, name=None)
        # 使用shuffle_batch可以随机打乱输入, capacity就是此队列的容量,min_after_dequeue的值一定要比capacity要小
        # 使用shuffle_batch的前提是图片的shape都都一样,本例下是不能使用的
        # img_data, shape, bboxs, bboxs_shape = tf.train.shuffle_batch([img_data, shape, bboxs, bboxs_shape],
        #                                                              batch_size=1, capacity=2000,
        #                                                              min_after_dequeue=1000)

        init = tf.global_variables_initializer()
        tf.train.match_filenames_once(tfrecord_path)

        im_fns = os.listdir(tfrecord_path)  # 图片目录
        im_fns = [tfrecord_path + os.sep + _ for _ in im_fns]
        im_nums = [GeneratorTfrecordTool.total_sample(_) for _ in im_fns]
        num = sum(im_nums)
        print("num:", num)
        coord = tf.train.Coordinator()
        with tf.Session() as sess:
            sess.run(init)
            queue_runner = tf.train.start_queue_runners(sess, coord=coord)
            for i in range(num*10):
                print("check:", i)
                # 必须放在循环体里面sess.run(数据op)
                img_data_, shape_, lable_seq_, lable_seq_len_, image_w_, lable_seq_dense_ = sess.run(
                    [img_data, shape, lable_seq, lable_seq_len, image_w, lable_seq_dense])

                lable_seq_ = tf.reshape(lable_seq_, [1, lable_seq_len.eval()])
                # do trian :feed_dict = {img_data_:img_data_....}
                print("type img_data:", type(img_data_))
                print("img_data_:", img_data_)
                print(img_data_.shape, shape_, lable_seq_.shape, lable_seq_len_, image_w_)
                print("lable_seq_:", lable_seq_.eval())
                lable_text = [lable_dic_opposite[_] for _ in lable_seq_.eval()[0]]
                print("".join(lable_text))
                print("lable_seq_dense_:", lable_seq_dense_)
                lable_seq_dense_text = [lable_dic_opposite[_] for _ in lable_seq_dense_[0]]
                print("".join(lable_seq_dense_text))
            coord.request_stop()
            coord.join(queue_runner)

    @staticmethod
    def read_and_decode_gfile_synthetic_chinese_string_dataset_tfrecord_data_and_show(tfrecord_name):
        global lable_dic_opposite
        if lable_dic_opposite is None:
            lable_dic_opposite = gen_and_get_lable_set.read_lable_dict_opposite()

        def _parse_record(example_proto):
            features_ = {
                'image/image_info': tf.FixedLenFeature([3], tf.int64),  # tf.int64,必须指定长度:3
                'image/lable_seq_len': tf.FixedLenFeature([1], tf.int64),  # tf.int64,必须指定长度:2
                'image/encoded': tf.FixedLenFeature([], tf.string),
                'image/lable_seq': tf.FixedLenFeature([], tf.string),
                'image/image_w':  tf.FixedLenFeature([1], tf.int64)  # image_w
            }
            parsed_features = tf.parse_single_example(example_proto, features=features_)
            return parsed_features

        # 用 dataset 读取 tfrecord 文件g
        dataset = tf.data.TFRecordDataset(tfrecord_name)
        dataset = dataset.map(_parse_record)
        iterator = dataset.make_one_shot_iterator()
        num = GeneratorTfrecordTool.total_sample(tfrecord_name)
        with tf.Session() as sess:
            for i in range(num):
                features = sess.run(iterator.get_next())
                img_data = features['image/encoded']
                shape = features['image/image_info']
                # print(type(shape), shape)
                lable_seq_len = features['image/lable_seq_len']
                # print(type(lable_seq_len), lable_seq_len)
                lable_seq = features['image/lable_seq']
                lable_seq = tf.decode_raw(lable_seq, tf.int64)
                lable_seq = tf.reshape(lable_seq, [1, lable_seq_len])
                lable_seqlist = lable_seq.eval()
                image_w = features['image/image_w']
                # 恢复为.jpg文件
                # tf.gfile.GFile(str(i) + '.jpg', 'wb').write(img_data)

                img_data = tf.image.decode_jpeg(img_data)
                # img_data = tf.decode_raw(img_data, tf.uint8)
                # image_data = np.reshape(img_data, shape)
                img_data = tf.reshape(img_data, shape)
                img_data = tf.cast(img_data, tf.float32) / 255.  # 必须除以255.

                plt.figure()
                # 显示图片
                plt.imshow(img_data.eval())

                # print(lable_seqlist[0], len(lable_seqlist[0]), shape)
                lable_text = [lable_dic_opposite[_] for _ in lable_seqlist[0]]
                print("".join(lable_text))
                print("image_w:", image_w)
                plt.show()

                # 将数据重新编码成 jpg 图片并保存
                # img_data = features['image/encoded']
                # tf.gfile.GFile(str(i) + '.jpg', 'wb').write(img_data)


def dense2sparse(arr_tensor):
    if isinstance(arr_tensor, tf.SparseTensor):
        return arr_tensor
    arr_idx = tf.where(tf.not_equal(arr_tensor, -1))
    # print("---arr_idx:", arr_idx, type(arr_idx),  tf.shape(arr_tensor))
    # tf.gather_nd:取出arr_tensor中对应索引为arr_idx的数据
    # arr_sparse = tf.SparseTensor(arr_idx, tf.gather_nd(arr_tensor, arr_idx), arr_tensor.get_shape())
    arr_sparse = tf.SparseTensor(arr_idx, tf.gather_nd(arr_tensor, arr_idx), tf.cast(tf.shape(arr_tensor), tf.int64))
    # print("---arr_sparse:", arr_sparse, type(arr_sparse))
    # arr_dense = tf.sparse_to_dense(arr_sparse.indices, arr_sparse.dense_shape, arr_sparse.values)
    return arr_sparse


if __name__ == '__main__':
    """
    GeneratorTfrecordTool.convert_to_tfrecord_data_gfile(
        "/Users/sherry/work/pycharm_python/text-detection-ctpn/text-detection-ctpn-banjin-dev/data/test",
        "/Users/sherry/work/pycharm_python/generator_tfrecord_data/g_train.tfrecord")
    nums = GeneratorTfrecordTool.total_sample(
        "/Users/sherry/work/pycharm_python/generator_tfrecord_data/g_train.tfrecord")
    print(nums)
    GeneratorTfrecordTool.read_and_decode_gfile_tfrecord_data_and_show(
        "/Users/sherry/work/pycharm_python/generator_tfrecord_data/g_train.tfrecord")

    GeneratorTfrecordTool.convert_to_tfrecord_data(
        "/Users/sherry/work/pycharm_python/text-detection-ctpn/text-detection-ctpn-banjin-dev/data/test",
        "/Users/sherry/work/pycharm_python/generator_tfrecord_data/train.tfrecord")
    nums = GeneratorTfrecordTool.total_sample(
        "/Users/sherry/work/pycharm_python/generator_tfrecord_data/train.tfrecord")
    print(nums)
    GeneratorTfrecordTool.read_and_decode_tfrecord_data_and_show(
        "/Users/sherry/work/pycharm_python/generator_tfrecord_data/train.tfrecord")
    gpu_id = 1
    print('/gpu:%d' % gpu_id)
    GeneratorTfrecordTool.read_gfile_tfrecord_data_sess(
        "/Users/sherry/work/pycharm_python/generator_tfrecord_data/g_train.tfrecord")

    GeneratorTfrecordTool.convert_to_tfrecord_data_gfile_synthetic_chinese_string_dataset(
        "/Users/sherry/data/Synthetic_Chinese_String_Dataset_224",
        "/Users/sherry/data/Synthetic_Chinese_String_Dataset_label",
        "/Users/sherry/work/pycharm_python/generator_tfrecord_data/gscsd_train.tfrecord")
    nums = GeneratorTfrecordTool.total_sample(
        "/Users/sherry/work/pycharm_python/generator_tfrecord_data/gscsd_train.tfrecord")
    print(nums)

    GeneratorTfrecordTool.read_and_decode_gfile_synthetic_chinese_string_dataset_tfrecord_data_and_show(
        "/Users/sherry/work/pycharm_python/generator_tfrecord_data/gscsd_train.tfrecord"
    )

    GeneratorTfrecordTool.read_gfile_synthetic_chinese_string_dataset_tfrecord_data_sess(
        "/Users/sherry/work/pycharm_python/generator_tfrecord_data/gscsd_train.tfrecord"
    )


    GeneratorTfrecordTool.convert_to_tfrecord_data_slice_gfile_synthetic_chinese_string_dataset(
        "/Users/sherry/data/Synthetic_Chinese_String_Dataset_224",
        "/Users/sherry/data/Synthetic_Chinese_String_Dataset_label",
        "/Users/sherry/work/pycharm_python/generator_tfrecord_data/data",
        "gscsd_train_x.tfrecord")
    GeneratorTfrecordTool.read_slice_gfile_synthetic_chinese_string_dataset_tfrecord_data_sess(
        "/Users/sherry/work/pycharm_python/generator_tfrecord_data/data"
    )
    """
    GeneratorTfrecordTool.convert_to_tfrecord_data_slice_gfile_synthetic_chinese_string_dataset(
        "/home/gaofangjie/Synthetic_Chinese_String_Dataset_224",
        "/home/gaofangjie/Synthetic_Chinese_String_Dataset_label",
        "/home/gaofangjie/data",
        "gscsd_train_x.tfrecord")

    nums = GeneratorTfrecordTool.total_sample_by_path("/home/gaofangjie/data")
    print(nums)
    print("done......")
