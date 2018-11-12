# -*-coding: utf-8 -*-
"""
    @Project: nlp-learning-tutorials
    @File   : create_label.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-10-27 11:54:19
"""
import utils.load_data as load_data
import os
def add_label(files_list,label):
    all_data=[]
    for file in files_list:
        all_data.append([file]+[label])
    return all_data

if __name__=="__main__":
    # maxSeqLength = 250
    # # 载入正负样本数据
    positiveFiles = load_data.load_file_list('../training_data/positiveReviews')
    negativeFiles = load_data.load_file_list('../training_data/negativeReviews')
    print("pos nums:{},neg nums:{}".format(len(positiveFiles), len(negativeFiles)))

    # positiveFiles=[os.path.basename(i)for i in positiveFiles]
    files_labels1=add_label(positiveFiles,label=1)

    # negativeFiles=[os.path.basename(i)for i in negativeFiles]
    files_labels2=add_label(negativeFiles,label=0)
    files_name='./labels.txt'
    files_labels=files_labels1+files_labels2
    load_data.save_file_label(files_labels,files_name)

    images_list, labels_list=load_data.load_file_label(files_name)
