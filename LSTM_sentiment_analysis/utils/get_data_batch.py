# -*-coding: utf-8 -*-
"""
    @Project: nlp-learning-tutorials
    @File   : get_data_batch.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-10-29 09:25:34
"""

import tensorflow as tf
import numpy as np
import  os

def get_data_batch(inputs, batch_size, labels_nums, one_hot=False, shuffle=False, num_threads=1):
    '''
    :param inputs: 输入数据，可以是多个list
    :param batch_size:
    :param labels_nums:标签个数
    :param one_hot:是否将labels转为one_hot的形式
    :param shuffle:是否打乱顺序,一般train时shuffle=True,验证时shuffle=False
    :return:返回batch的images和labels
    '''
    # 生成队列
    inputs_que = tf.train.slice_input_producer(inputs, shuffle=shuffle)
    min_after_dequeue = 200
    capacity = min_after_dequeue + 3 * batch_size  # 保证capacity必须大于min_after_dequeue参数值
    if shuffle:
        out_batch = tf.train.shuffle_batch(inputs_que,
                                           batch_size=batch_size,
                                           capacity=capacity,
                                           min_after_dequeue=min_after_dequeue,
                                           num_threads=num_threads)
    else:
        out_batch = tf.train.batch(inputs_que,
                                   batch_size=batch_size,
                                   capacity=capacity,
                                   num_threads=num_threads)
    return out_batch


def get_batch_images(images, labels, batch_size, labels_nums, one_hot=False, shuffle=False):
    '''
    :param images:图像
    :param labels:标签
    :param batch_size:
    :param labels_nums:标签个数
    :param one_hot:是否将labels转为one_hot的形式
    :param shuffle:是否打乱顺序,一般train时shuffle=True,验证时shuffle=False
    :return:返回batch的images和labels
    '''
    images_que, labels_que = tf.train.slice_input_producer([images, labels], shuffle=shuffle)
    min_after_dequeue = 200
    capacity = min_after_dequeue + 3 * batch_size  # 保证capacity必须大于min_after_dequeue参数值
    if shuffle:
        images_batch, labels_batch = tf.train.shuffle_batch([images_que, labels_que],
                                                            batch_size=batch_size,
                                                            capacity=capacity,
                                                            min_after_dequeue=min_after_dequeue)
    else:
        images_batch, labels_batch = tf.train.batch([images_que, labels_que],
                                                    batch_size=batch_size,
                                                    capacity=capacity)
    if one_hot:
        labels_batch = tf.one_hot(labels_batch, labels_nums, 1, 0)
    return images_batch, labels_batch


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

def load_indexMat(filename,squeeze=True):
    indexMat = np.load(filename)
    label_path=os.path.join(os.path.dirname(filename),os.path.basename(filename).split('.')[0])+'_labels.npy'
    labels =np.load(label_path)
    if squeeze:
        labels=np.squeeze(labels)
    return indexMat,labels

if __name__ == '__main__':
    out_train_path = '../training_data/train_indexMat.npy'
    out_val_path = '../training_data/val_indexMat.npy'
    indexMat, labels=load_indexMat(out_train_path,True)
    # indexMat = np.reshape(np.arange(8*3), (8,3))
    # labels = np.reshape(np.arange(8*1), (8,1))
    # indexMat=indexMat.tolist()
    # labels=labels.tolist()
    batch_images, batch_labels = get_batch_images(indexMat, labels, batch_size=5, labels_nums=2, one_hot=True,
                                                  shuffle=False)
    iter = 5  # 迭代5次，每次输出一个batch个
    with tf.Session() as sess:  # 开始一个会话
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(iter):
            # 在会话中取出images和labels
            images, labels = sess.run([batch_images, batch_labels])
            print('**************************')
            print('batch_images shape:{},batch_labels shape:{}'.format(images.shape,labels.shape))
            print('batch_images:{}'.format(images))
            print('batch_labels:{}'.format(labels))

        # 停止所有线程
        coord.request_stop()
        coord.join(threads)

