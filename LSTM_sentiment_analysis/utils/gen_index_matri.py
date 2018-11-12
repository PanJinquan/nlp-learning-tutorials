# -*-coding: utf-8 -*-
"""
    @Project: LSTM
    @File   : gen_index_matri.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-10-21 16:00:58
    @info :  将文件中的文本转换成索引矩阵:
    基本过程：
    -> split word 分词
    -> id=wordsList.index(word) 根据wordsList获得id，所以文本的id构成索引矩阵indexMat
    -> vector=wordVectors[firstSentence]或者tf.nn.embedding_lookup(wordVectors, id) 转化为wordVectors
"""
import re
import numpy as np
import random
from utils.load_wordVectors import *
import utils.load_data as load_data
import TxtStorage as TxtStorage
import os

def cleanSentences(string):
    '''
    删除文本的标点符号、括号、问号等，只留下字母数字字符
    :param string:
    :return:
    '''
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

def gen_indexMat(file_list,wordsList_path,wordVectors_path,maxSeqLength):
    '''
    所以文本的id构成索引矩阵indexMat：
    将文件中的文本转换成索引矩阵， 25000 条评论，得到一个 25000 * 250 的矩阵
    其中25000是文本个数，250是每个文本读取250单词，并转换为下标索引,不足的补0，多余的不要了
    :param file_list:
    :return:
    '''
    wordsList, wordVectors = load_wordVectors(wordsList_path, wordVectors_path)
    numFiles=len(file_list)
    # 将单词转换成索引矩阵
    indexMat = np.zeros((numFiles, maxSeqLength), dtype='int32')
    fileCounter = 0
    for pf in file_list:
        with open(pf, "r",encoding='UTF-8') as f:
            indexCounter = 0
            line = f.readline() # 仅读取一行
            # line = f.readline() # 读取所有行内容
            cleanedLine = cleanSentences(line)
            split = cleanedLine.split()
            for word in split:
                try:
                    indexMat[fileCounter][indexCounter] = wordsList.index(word)
                except ValueError:
                    indexMat[fileCounter][indexCounter] = 399999  # Vector for unkown words
                indexCounter = indexCounter + 1
                if indexCounter >= maxSeqLength:
                    break
            fileCounter = fileCounter + 1
            if fileCounter%100 == 0:
                print("step:{},filename:{}".format(fileCounter,pf))
    return indexMat


def create_tain_val_indexMat(filename_dir,filename,
                             wordsList_path,wordVectors_path,
                             maxSeqLength,
                             out_train_path,out_val_path):
    files_list, labels_list=load_data.load_file_label(filename)
    print("files_list nums:{}".format(len(files_list)))
    files_list=[ os.path.join(filename_dir,name) for name in files_list]
    shuffle=True
    if shuffle:
        # seeds = random.randint(0,len(files_list)) #产生一个随机数种子
        seeds = 100 # 固定种子,只要seed的值一样，后续生成的随机数都一样
        random.seed(seeds)
        random.shuffle(files_list)
        random.seed(seeds)
        random.shuffle(labels_list)

    ratio=0.5 # train数据的占比
    split_indnex=int(len(files_list)*ratio)
    # train文本数据
    train_list=files_list[:split_indnex]
    train_label=labels_list[:split_indnex]
    # val文本数据
    val_list=files_list[split_indnex:]
    val_label=labels_list[split_indnex:]

    # 获得train，val文本的单词索引矩阵
    train_indexMat=gen_indexMat(train_list,wordsList_path,wordVectors_path,maxSeqLength)
    val_indexMat=gen_indexMat(val_list,wordsList_path,wordVectors_path,maxSeqLength)

    # 保存train，val文本的单词索引矩阵
    np.save(out_train_path, train_indexMat)
    np.save(out_val_path, val_indexMat)
    # 保存train，val文本的label
    out_train_label=os.path.join(os.path.dirname(out_train_path),os.path.basename(out_train_path).split('.')[0])+'_labels.npy'
    out_val_label=os.path.join(os.path.dirname(out_val_path),os.path.basename(out_val_path).split('.')[0])+'_labels.npy'
    np.save(out_train_label, train_label)
    np.save(out_val_label, val_label)

def load_indexMat(filename):
    indexMat = np.load(filename)
    label_path=os.path.join(os.path.dirname(filename),os.path.basename(filename).split('.')[0])+'_labels.npy'
    labels = np.load(label_path)
    return indexMat,labels

def create_indexMat():
    wordsList_path = '../training_data/wordsList.npy'
    wordVectors_path = '../training_data/wordVectors.npy'
    maxSeqLength = 250
    # # 载入正负样本数据
    positiveFiles = load_data.load_file_list('../training_data/positiveReviews')
    negativeFiles = load_data.load_file_list('../training_data/negativeReviews')
    print("pos nums:{},neg nums:{}".format(len(positiveFiles), len(negativeFiles)))
    files_list = positiveFiles + negativeFiles
    indexMat=gen_indexMat(files_list,wordsList_path,wordVectors_path,maxSeqLength)
    np.save('indexMat', indexMat)

if __name__=="__main__":
    wordsList_path='../training_data/wordsList.npy'
    wordVectors_path='../training_data/wordVectors.npy'
    maxSeqLength = 250
    out_train_path = '../training_data/train_indexMat.npy'
    out_val_path = '../training_data/val_indexMat.npy'
    filename_dir='../training_data'
    # filename='../training_data/test.txt'
    filename='../training_data/train_val.txt'
    # create_tain_val_indexMat(filename_dir,
    #                          filename,
    #                          wordsList_path,
    #                          wordVectors_path,
    #                          maxSeqLength,
    #                          out_train_path,
    #                          out_val_path)
    indexMat, labels = load_indexMat(out_train_path)
