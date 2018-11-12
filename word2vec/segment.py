# -*-coding: utf-8 -*-
"""
    @Project: nlp-learning-tutorials
    @File   : segment.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-11-11 17:51:53
"""

##
import jieba
import os
from utils import files_processing

'''
read() 每次读取整个文件，它通常将读取到底文件内容放到一个字符串变量中，也就是说 .read() 生成文件内容是一个字符串类型。
readline()每只读取文件的一行，通常也是读取到的一行内容放到一个字符串变量中，返回str类型。
readlines()每次按行读取整个文件内容，将读取到的内容放到一个列表中，返回list类型。
'''
def getStopwords(path):
    '''
    加载停用词
    :param path:
    :return:
    '''
    stopwords = []
    with open(path, "r", encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            stopwords.append(line.strip())
    return stopwords

def segment_line(file_list,segment_out_dir,stopwords=[]):
    '''
    字词分割，对每行进行字词分割
    :param file_list:
    :param segment_out_dir:
    :param stopwords:
    :return:
    '''
    for i,file in enumerate(file_list):
        segment_out_name=os.path.join(segment_out_dir,'segment_{}.txt'.format(i))
        segment_file = open(segment_out_name, 'a', encoding='utf8')
        with open(file, encoding='utf8') as f:
            text = f.readlines()
            for sentence in text:
                # jieba.cut():参数sentence必须是str(unicode)类型
                sentence = list(jieba.cut(sentence))
                sentence_segment = []
                for word in sentence:
                    if word not in stopwords:
                        sentence_segment.append(word)
                segment_file.write(" ".join(sentence_segment))
            del text
            f.close()
        segment_file.close()

def segment_lines(file_list,segment_out_dir,stopwords=[]):
    '''
    字词分割，对整个文件内容进行字词分割
    :param file_list:
    :param segment_out_dir:
    :param stopwords:
    :return:
    '''
    for i,file in enumerate(file_list):
        segment_out_name=os.path.join(segment_out_dir,'segment_{}.txt'.format(i))
        with open(file, 'rb') as f:
            document = f.read()
            # document_decode = document.decode('GBK')
            document_cut = jieba.cut(document)
            sentence_segment=[]
            for word in document_cut:
                if word not in stopwords:
                    sentence_segment.append(word)
            result = ' '.join(sentence_segment)
            result = result.encode('utf-8')
            with open(segment_out_name, 'wb') as f2:
                f2.write(result)


if __name__=='__main__':


    # 多线程分词
    # jieba.enable_parallel()
    # 加载自定义词典
    user_path = 'data/user_dict.txt'
    jieba.load_userdict(user_path)

    stopwords_path='data/stop_words.txt'
    stopwords=getStopwords(stopwords_path)

    file_dir='data/source'
    segment_out_dir='data/segment'
    file_list=files_processing.get_files_list(file_dir,postfix='*.txt')
    segment_lines(file_list, segment_out_dir,stopwords)
