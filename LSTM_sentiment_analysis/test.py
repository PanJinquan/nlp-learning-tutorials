# -*-coding: utf-8 -*-
"""
    @Project: create_batch_data
    @File   : create_batch_data.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2017-10-27 18:20:15
"""
import math
import random
import os
import glob
import numpy as np


def get_data_batch(inputs, batch_size=None, shuffle=False):
    '''
    循环产生批量数据batch
    :param inputs: list类型数据，多个list,请[list0,list1,...]
    :param batch_size: batch大小
    :param shuffle: 是否打乱inputs数据
    :return: 返回一个batch数据
    '''
    rows = len(inputs[0])
    indices = list(range(rows))
    # 如果输入是list,则需要转为list
    if shuffle:
        random.seed(100)
        random.shuffle(indices)
    while True:
        batch_indices = np.asarray(indices[0:batch_size])  # 产生一个batch的index
        indices = indices[batch_size:] + indices[:batch_size]  # 循环移位，以便产生下一个batch
        batch_data = []
        for data in inputs:
            data = np.asarray(data)
            temp_data=data[batch_indices] #使用下标查找，必须是ndarray类型类型
            batch_data.append(temp_data.tolist())
        yield batch_data

def get_data_batch2(inputs, batch_size=None, shuffle=False):
    '''
    循环产生批量数据batch
    :param inputs: list类型数据，多个list,请[list0,list1,...]
    :param batch_size: batch大小
    :param shuffle: 是否打乱inputs数据
    :return: 返回一个batch数据
    '''
    # rows,cols=inputs.shape
    rows = len(inputs[0])
    indices = list(range(rows))
    if shuffle:
        random.seed(100)
        random.shuffle(indices)
    while True:
        batch_indices = indices[0:batch_size]  # 产生一个batch的index
        indices = indices[batch_size:] + indices[:batch_size]  # 循环移位，以便产生下一个batch
        batch_data = []
        for data in inputs:
            temp_data = find_list(batch_indices, data)
            batch_data.append(temp_data)
        yield batch_data


def find_list(indices, data):
    out = []
    for i in indices:
        out = out + [data[i]]
    return out


def get_list_batch(inputs, batch_size=None, shuffle=False):
    '''
    循环产生batch数据
    :param inputs: list数据
    :param batch_size: batch大小
    :param shuffle: 是否打乱inputs数据
    :return: 返回一个batch数据
    '''
    if shuffle:
        random.shuffle(inputs)
    while True:
        batch_inouts = inputs[0:batch_size]
        inputs = inputs[batch_size:] + inputs[:batch_size]  # 循环移位，以便产生下一个batch
        yield batch_inouts


def load_file_list(text_dir):
    text_dir = os.path.join(text_dir, '*.txt')
    text_list = glob.glob(text_dir)
    return text_list


def get_next_batch(batch):
    return batch.__next__()


def load_image_labels(finename):
    '''
    载图txt文件，文件中每行为一个图片信息，且以空格隔开：图像路径 标签1 标签1，如：test_image/1.jpg 0 2
    :param test_files:
    :return:
    '''
    images_list = []
    labels_list = []
    with open(finename) as f:
        lines = f.readlines()
        for line in lines:
            # rstrip：用来去除结尾字符、空白符(包括\n、\r、\t、' '，即：换行、回车、制表符、空格)
            content = line.rstrip().split(' ')
            name = content[0]
            labels = []
            for value in content[1:]:
                labels.append(float(value))
            images_list.append(name)
            labels_list.append(labels)
    return images_list, labels_list


if __name__ == '__main__':
    filename = './training_data/test.txt'
    images_list, labels_list = load_image_labels(filename)

    # 若输入为np.arange数组，则需要tolist()为list类型，如：
    # images_list = np.reshape(np.arange(8*3), (8,3))
    # labels_list = np.reshape(np.arange(8*3), (8,3))
    # images_list=images_list.tolist()
    # labels_list=labels_list.tolist()

    iter = 5  # 迭代5次，每次输出一个batch个
    # batch = get_data_batch([images_list, labels_list], batch_size=3, shuffle=False)
    batch = get_data_batch(inputs=[images_list,labels_list], batch_size=5, shuffle=True)

    for i in range(iter):
        print('**************************')
        batch_images, batch_labels = get_next_batch(batch)
        print('batch_images:{}'.format(batch_images))
        print('batch_labels:{}'.format(batch_labels))


