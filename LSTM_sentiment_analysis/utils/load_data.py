# -*-coding: utf-8 -*-
"""
    @Project: LSTM
    @File   : load_data.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-10-21 10:26:54
"""
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import re
import random
from utils.load_wordVectors import *

def load_file_list(text_dir):
    text_dir=os.path.join(text_dir,'*.txt')
    text_list=glob.glob(text_dir)
    return text_list

def merge_list(data1, data2):
    '''
    将两个list进行合并
    :param data1:
    :param data2:
    :return:返回合并后的list
    '''
    if not len(data1) == len(data2):
        return
    all_data = []
    for d1, d2 in zip(data1, data2):
        all_data.append(d1 + d2)
    return all_data


def split_list(data, split_index=1):
    '''
    将data切分成两部分
    :param data: list
    :param split_index: 切分的位置
    :return:
    '''
    data1 = []
    data2 = []
    for d in data:
        d1 = d[0:split_index]
        d2 = d[split_index:]
        data1.append(d1)
        data2.append(d2)
    return data1, data

def load_file_label(filename):
    '''
    载图txt文件，文件中每行为一个图片信息，且以空格隔开：图像路径 标签1 标签1，如：test_image/1.txt 0 2
    :param test_files:
    :return:
    '''
    images_list=[]
    labels_list=[]
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            #rstrip：用来去除结尾字符、空白符(包括\n、\r、\t、' '，即：换行、回车、制表符、空格)
            content=line.rstrip().split(' ')
            name=content[0]
            labels=[]
            for value in content[1:]:
                labels.append(int(value))
            images_list.append(name)
            labels_list.append(labels)
    return images_list,labels_list

def save_file_label(content, filename, mode='w'):
    """保存txt数据
            :param content:需要保存的数据,type->list
            :param filename:文件名
            :param mode:读写模式:'w' or 'a'
            :return: void
            """
    with open(filename, mode) as f:
        for line in content:
            str_line = ""
            for col, data in enumerate(line):
                if not col == len(line) - 1:
                    # 以空格作为分隔符
                    str_line = str_line + str(data) + " "
                else:
                    # 每行最后一个数据用换行符“\n”
                    str_line = str_line + str(data) + "\n"
            f.write(str_line)

def read_text_data(file_list):
    '''
    :return:
    '''
    contents = []
    for pf in file_list:
        with open(pf, "r", encoding='utf-8') as f:
            line = f.readline()#仅读取一行
            # line = f.readlines()#读取多行
            contents.append(line)
    return contents

def statistics_text_info(file_list):
    '''
    统计每个文本有多少个单词
    :return:
    '''
    numWords = []
    for pf in file_list:
        with open(pf, "r", encoding='utf-8') as f:
            # line = f.readlines()#读取多行
            line = f.readline()
            counter = len(line.split())
            numWords.append(counter)
    return  numWords

def show_data_info(file_list):
    '''
    The total number of files is 25000
    The total number of words in the files is 5844680
    The average number of words in the files is 233.7872
    :param numWords:
    :return:
    '''
    # 从直方图和句子的平均单词数，我们认为将句子最大长度设置为 250 是可行的。
    numWords=statistics_text_info(file_list)
    numFiles = len(numWords)
    print('The total number of files is', numFiles)
    print('The total number of words in the files is', sum(numWords))
    print('The average number of words in the files is', sum(numWords) / len(numWords))
    plt.hist(numWords, 50)
    plt.xlabel('Sequence Length')
    plt.ylabel('Frequency')
    plt.axis([0, 1200, 0, 8000])
    plt.show()

def get_text_batch(inputs, wordsList_path, wordVectors_path,batch_size=None,maxSeqLength = 250,shuffle=False):
    '''
    循环产生batch数据
    :param inputs: list数据
    :param batch_size: batch大小
    :param shuffle: 是否打乱inputs数据
    :return: 返回一个batch数据
    '''
    wordsList, wordVectors = load_wordVectors(wordsList_path, wordVectors_path)
    if shuffle:
        random.shuffle(inputs)
    while True:
        batch_inouts = inputs[0:batch_size]
        inputs=inputs[batch_size:] + inputs[:batch_size]# 循环移位，以便产生下一个batch
        matri=gen_index_matri(batch_inouts,wordsList, wordVectors,maxSeqLength)
        yield matri

def cleanSentences(string):
    '''
    删除文本的标点符号、括号、问号等，只留下字母数字字符
    :param string:
    :return:
    '''
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

def gen_index_matri(file_list,wordsList, wordVectors,maxSeqLength):
    '''
    将文件中的文本转换成索引矩阵， 25000 条评论，得到一个 25000 * 250 的矩阵
    :param file_list:
    :return:
    '''
    numFiles=len(file_list)
    index_matri = np.zeros((numFiles, maxSeqLength), dtype='int32')
    fileCounter = 0
    for pf in file_list:
        with open(pf, "r") as f:
            indexCounter = 0
            line = f.readline() # 仅读取一行
            # line = f.readline() # 读取所有行内容
            cleanedLine = cleanSentences(line)
            split = cleanedLine.split()
            for word in split:
                try:
                    index_matri[fileCounter][indexCounter] = wordsList.index(word)
                except ValueError:
                    index_matri[fileCounter][indexCounter] = 399999  # Vector for unkown words
                indexCounter = indexCounter + 1
                if indexCounter >= maxSeqLength:
                    break
            fileCounter = fileCounter + 1
    return index_matri


if __name__=='__main__':
    inputs = load_file_list('../training_data/test')
    wordsList_path = '../training_data/wordsList.npy'
    wordVectors_path = '../training_data/wordVectors.npy'
    iter = 2  # 迭代10次，每次输出5个
    batch = get_text_batch(inputs,wordsList_path,wordVectors_path, batch_size=5, shuffle=False)
    for i in range(iter):
        print('**************************')
        print(batch.__next__())