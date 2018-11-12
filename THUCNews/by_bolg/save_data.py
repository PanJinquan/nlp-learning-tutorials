# -*-coding: utf-8 -*-
"""
    @Project: nlp-learning-tutorials
    @File   : gen_data.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-10-29 17:31:34
"""
import os

# 获取文本文件路径
def getFilePathList(rootDir):
    filePath_list = []
    for walk in os.walk(rootDir):
        part_filePath_list = [os.path.join(walk[0], file) for file in walk[2]]
        filePath_list.extend(part_filePath_list)
    return filePath_list
THUCNews_path="/home/ubuntu/project/tfTest/THUCNews/test"
filePath_list = getFilePathList(THUCNews_path)
print(len(filePath_list))

# 获取所有样本标签
label_list = []
for filePath in filePath_list:
    label = filePath.split('/')[-2]
    label_list.append(label)
print(len(label_list))

# 标签统计计数
import pandas as pd
print(pd.value_counts(label_list))

# 调用pickle库保存label_list
import pickle
with open('label_list.pickle', 'wb') as file:
    pickle.dump(label_list, file)

# 获取所有样本内容、保存content_list
import time
import pickle
import re
import io

def getFile(filePath):
    with io.open(filePath, encoding='utf8') as file:
        fileStr = ''.join(file.readlines(1000))
    return fileStr

interval = 20000
n_samples = len(label_list)
startTime = time.time()
# directory_name = 'content_list'
directory_name = 'test'
if not os.path.isdir(directory_name):
    os.mkdir(directory_name)
for i in range(0, n_samples, interval):
    startIndex = i
    endIndex = i + interval
    content_list = []
    print('%06d-%06d start' %(startIndex, endIndex))
    for filePath in filePath_list[startIndex:endIndex]:
        fileStr = getFile(filePath)
        content = re.sub('\s+', ' ', fileStr)
        content_list.append(content)
    save_fileName = directory_name + '/%06d-%06d.pickle' %(startIndex, endIndex)
    with open(save_fileName, 'wb') as file:
        pickle.dump(content_list, file)
    used_time = time.time() - startTime
    print('%06d-%06d used time: %.2f seconds' %(startIndex, endIndex, used_time))
