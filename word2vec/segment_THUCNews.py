# -*-coding: utf-8 -*-
"""
    @Project: nlp-learning-tutorials
    @File   : segment.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2017-05-11 17:51:53
"""
import jieba
import os
import io
import math
import re
from utils import files_processing,segment

if __name__=='__main__':


    # 多线程分词
    # jieba.enable_parallel()
    # 加载自定义词典
    user_path = './data/user_dict.txt'
    jieba.load_userdict(user_path)

    # stopwords_path='data/stop_words.txt'
    # stopwords=load_stopwords(stopwords_path)
    stopwords=segment.common_stopwords()

    file_dir='./data/source'
    segment_out_dir='./data/segment'
    files_list=files_processing.get_files_list(file_dir,postfix='*.txt')

    # segment_out_dir='data/segment_conbine.txt'
    # combine_files_content(files_list, segment_out_dir,stopwords)
    segment.batch_processing_files(files_list, segment_out_dir, batchSize=1000, stopwords=[])
