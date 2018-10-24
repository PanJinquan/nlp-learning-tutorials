# -*-coding: utf-8 -*-
"""
    @Project: LSTM
    @File   : gen_index_matri.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-10-21 16:00:58
    @info :  将文件中的文本转换成索引矩阵
"""
import re
import numpy as np
import load_data

def cleanSentences(string):
    '''
    删除文本的标点符号、括号、问号等，只留下字母数字字符
    :param string:
    :return:
    '''
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

def gen_index_matri(file_list,wordsList_path,wordVectors_path,maxSeqLength):
    '''
    将文件中的文本转换成索引矩阵， 25000 条评论，得到一个 25000 * 250 的矩阵
    :param file_list:
    :return:
    '''
    wordsList, wordVectors = load_data.load_wordVectors(wordsList_path, wordVectors_path)
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
if __name__=="__main__":
    maxSeqLength = 250
    # 载入正负样本数据
    # positiveFiles = load_data.load_file_list('./training_data/positiveReviews')
    # negativeFiles = load_data.load_file_list('./training_data/negativeReviews')
    positiveFiles = load_data.load_file_list('./training_data/posTest')
    negativeFiles = load_data.load_file_list('./training_data/negTest')
    print("pos nums:{},neg nums:{}".format(len(positiveFiles), len(negativeFiles)))


    # posData = load_data.read_text_data(positiveFiles)
    # negData = load_data.read_text_data(negativeFiles)
    file_list=positiveFiles+negativeFiles

    wordsList_path = './training_data/wordsList.npy'
    wordVectors_path = './training_data/wordVectors.npy'
    index_matri=gen_index_matri(file_list,wordsList_path,wordVectors_path,maxSeqLength)

    #Pass into embedding function and see if it evaluates.
    np.save('my_idsMatrix', index_matri)
