# coding=utf-8
"""
    char to index map
    index to char map
    illool@163.com
"""
import os

lable_dic_file = "lable_set.txt"
path = r"/Users/sherry/data/Synthetic Chinese String Dataset label"
paths = os.listdir(path)
ocr_dict = set()


def read_line(path):
    global ocr_dict
    f = open(path)  # 返回一个文件对象
    line = f.readline()  # 调用文件的 readline()方法
    while line:
        # print(line)  # 在 Python 3 中使用
        string = line.split(" ")[-1]
        ocr_dict = ocr_dict.union(set(string))
        # print(ocr_dict)
        line = f.readline()
    f.close()


def read_all_file(paths):
    for file in paths:
        print(file)
        read_line(os.path.join(path, file))


def gen_dict():
    read_all_file(paths)
    print(os.path.join(path, "lable_set.txt"))
    with open(os.path.join(path, "lable_set.txt"), 'w') as file_object:
        for _ in ocr_dict:
            print(_)
            file_object.write(_ + '\n')


def read_lable_dict():
    with open(os.path.join(path, "lable_set.txt"), 'r', encoding='utf-8') as f:
        text = f.read()
        return text.split("\n")


if __name__ == '__main__':
    # gen_dict()
    lable_set = read_lable_dict()
    print(lable_set)
    print(len(lable_set))
