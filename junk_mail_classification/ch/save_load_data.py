# -*-coding: utf-8 -*-
"""
    @Project: nlp-learning-tutorials
    @File   : save_load_data.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-11-07 11:36:03
"""
import time
import pickle
import os
import io
import re
import numpy as np
from sklearn import preprocessing
import data_helpers
import word2vec_helpers

def getFilePathList(rootDir):
    '''
    获取rootDir下所有的文本文件的路径
    :param rootDir:
    :return:
    '''
    filePath_list = []
    for walk in os.walk(rootDir):
        part_filePath_list = [os.path.join(walk[0], file) for file in walk[2]]
        filePath_list.extend(part_filePath_list)
    return filePath_list

def gen_files_labels(files_dir):
    '''
    获取files_dir路径下所有文件路径，以及labels,其中labels用父级文件名表示
    files_dir目录下，同一类别的文件放一个文件夹，其labels即为文件的名
    :param files_dir:
    :return:filePath_list所有文件的路径,label_list对应的labels
    '''
    filePath_list = getFilePathList(files_dir)
    print("files nums:{}".format(len(filePath_list)))
    # 获取所有样本标签
    label_list = []
    for filePath in filePath_list:
        label = filePath.split(os.sep)[-2]
        label_list.append(label)

    labels_set=list(set(label_list))
    print("labels:{}".format(labels_set))
    # 标签统计计数
    import pandas as pd
    print(pd.value_counts(label_list))
    return filePath_list,label_list

def save_files_labels(files_list,out_files_dir,label_list,out_label_dir):
    '''
    保存文本数据以及labels
    :param files_list: 文件路径列表
    :param out_files_dir:输出保存文件内容的目录
    :param label_list: labels列表，与files_list一个对应
    :param out_label_dir:输出保存labels内容的目录
    :return: None
    '''
    if not os.path.exists(out_files_dir):
        os.makedirs(out_files_dir)

    if not os.path.exists(out_label_dir):
        os.makedirs(out_label_dir)

    # 获取所有样本内容、保存content_list
    interval = 20000
    n_samples = len(label_list)
    startTime = time.time()
    for i in range(0, n_samples, interval):
        startIndex = i
        endIndex = i + interval
        content_list = []
        print('%06d-%06d start' % (startIndex, endIndex))
        for filePath in files_list[startIndex:endIndex]:
            content = read_and_clean_zh_file(filePath)
            content_list.append(content)
        save_fileName = out_files_dir + '/%06d-%06d.pickle' % (startIndex, endIndex)
        with open(save_fileName, 'wb') as file:
            pickle.dump(content_list, file)
        used_time = time.time() - startTime
        print('%06d-%06d used time: %.2f seconds' % (startIndex, endIndex, used_time))

    # 调用pickle库保存label_list
    with open(os.path.join(out_label_dir,'label_list.pickle'), 'wb') as file:
        pickle.dump(label_list, file)


def clean_str(string):
    string = re.sub(r"[^\u4e00-\u9fff]", " ", string)
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    # string = re.sub(r"\'s", " \'s", string)
    # string = re.sub(r"\'ve", " \'ve", string)
    # string = re.sub(r"n\'t", " n\'t", string)
    # string = re.sub(r"\'re", " \'re", string)
    # string = re.sub(r"\'d", " \'d", string)
    # string = re.sub(r"\'ll", " \'ll", string)
    # string = re.sub(r",", " , ", string)
    # string = re.sub(r"!", " ! ", string)
    # string = re.sub(r"\(", " \( ", string)
    # string = re.sub(r"\)", " \) ", string)
    # string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    # return string.strip().lower()
    return string.strip()

def seperate_line(line):
    return ''.join([word + ' ' for word in line])

def read_and_clean_zh_file(input_file, one_line = False):
    '''
    :param input_file:
    :param one_line: true时，每个文件中的每行当作一个样本，Ｆalse时，以一个文件当作一个样本
    :return:
    '''
    if one_line:
        lines = list(open(input_file, "rb").readlines())
        content = [clean_str(seperate_line(line.decode('utf-8'))) for line in lines]
    else:
        with io.open(input_file, encoding='utf8') as file:
            lines = file.readlines()
            lines = ''.join(lines)
        content = clean_str(seperate_line(lines))
    #句子分割，空格隔开
    return content

def getFilePathList(rootDir):
    filePath_list = []
    for walk in os.walk(rootDir):
        part_filePath_list = [os.path.join(walk[0], file) for file in walk[2]]
        filePath_list.extend(part_filePath_list)
    return filePath_list

def  load_files_labels(content_dir, labels_dir, one_hot=False):
    '''
    加载文本内容和labels
    :param content_dir: 文本内容所在的目录
    :param labels_dir: labels所在的目录
    :param one_hot: 是否独热编码
    :return: content_list ,labels
    '''
    contentListPath_list = getFilePathList(content_dir)
    content_list = []
    for filePath in contentListPath_list:
        with open(filePath, 'rb') as file:
            part_content_list = pickle.load(file)
        content_list.extend(part_content_list)

    with open(os.path.join(labels_dir,'label_list.pickle'), 'rb') as file:
        label_list = pickle.load(file)

    # 将labels转为整数编码
    # labels_set=list(set(label_list))
    # labels=[]
    # for label in  label_list:
    #     for k in range(len(labels_set)):
    #         if label==labels_set[k]:
    #             labels+=[k]
    #             break
    # labels = np.asarray(labels)

    # 也可以用下面的方法：将labels转为整数编码
    labelEncoder = preprocessing.LabelEncoder()
    labels = labelEncoder.fit_transform(label_list)
    labels_set = labelEncoder.classes_

    for i in range(len(labels_set)):
        print("labels:{}->{}".format(labels_set[i],i))

    # 是否进行独热编码
    if one_hot:
        labels = labels.reshape(len(labels), 1)
        onehot_encoder = preprocessing.OneHotEncoder(sparse=False,categories='auto')
        labels = onehot_encoder.fit_transform(labels)
    return content_list ,labels

def save_data_vector(contents_dir, labels_dir,out_dir):
    x_text, y = load_files_labels(contents_dir, labels_dir, one_hot=True)
    # Get embedding vector, 句子padding到最大长度190，pandding内容为： '<PADDING>'
    sentences, max_document_length = data_helpers.padding_sentences(x_text, '<PADDING>', padding_sentence_length=190)
    embedding_dim=128
    x = np.array(word2vec_helpers.embedding_sentences(sentences, embedding_size=embedding_dim,
                                                      file_to_save=os.path.join(out_dir, 'trained_word2vec.model')))
    print("x.shape = {}".format(x.shape))  # shape=(10000, 190, 128)->(样本个数10000,每个样本的字词个数190，每个字词的向量长度128)
    print("y.shape = {}".format(y.shape))  # y.shape = (10000, 2)->样本的labels,以one-hot编码
    np.save(os.path.join(out_dir, 'data_vector.npy'),x)
    np.save(os.path.join(out_dir, 'labels.npy'),y)

def load_data_vector(data_path,labels_path):
    data_vector = np.load(data_path)
    labels = np.load(labels_path)
    return data_vector,labels

if __name__=="__main__":
    THUCNews_path = "D:/tensorflow/nlp-learning-tutorials/THUCNews/data"
    files_list, label_list = gen_files_labels(THUCNews_path)
    out_files_dir='contents_list'
    out_label_path='labels_list'
    # save_files_labels(files_list, out_files_dir, label_list, out_label_path)

    content_list, labels=load_files_labels(out_files_dir,out_label_path)
    sample_size = len(content_list)
    print('sample_size:{}'.format(sample_size))
    print('labels:{}'.format(labels))
