# -*-coding: utf-8 -*-
"""
    @Project: nlp-learning-tutorials
    @File   : word2vec_gensim.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2017-05-11 17:04:35
"""

from gensim.models import word2vec
import multiprocessing

def train_wordVectors(sentences, embedding_size = 128, window = 5, min_count = 5):
    '''

    :param sentences: sentences可以是LineSentence或者PathLineSentences读取的文件对象，也可以是
                    The `sentences` iterable can be simply a list of lists of tokens,如lists=[['我','是','中国','人'],['我','的','家乡','在','广东']]
    :param embedding_size: 词嵌入大小
    :param window: 窗口
    :param min_count:Ignores all words with total frequency lower than this.
    :return: w2vModel
    '''
    w2vModel = word2vec.Word2Vec(sentences, size=embedding_size, window=window, min_count=min_count,workers=multiprocessing.cpu_count())
    return w2vModel

def save_wordVectors(w2vModel,word2vec_path):
    w2vModel.save(word2vec_path)

def load_wordVectors(word2vec_path):
    w2vModel = word2vec.Word2Vec.load(word2vec_path)
    return w2vModel

if __name__=='__main__':

    # [1]若只有一个文件，使用LineSentence读取文件
    # segment_path='./data/segment/segment_0.txt'
    # sentences = word2vec.LineSentence(segment_path)

    # [1]若存在多文件，使用PathLineSentences读取文件列表

    segment_dir='./data/segment'
    sentences = word2vec.PathLineSentences(segment_dir)

    # 简单的训练
    model = word2vec.Word2Vec(sentences, hs=1,min_count=1,window=3,size=100)
    print(model.wv.similarity('沙瑞金', '高育良'))
    # print(model.wv.similarity('李达康'.encode('utf-8'), '王大路'.encode('utf-8')))

    # 一般训练，设置以下几个参数即可：
    word2vec_path='./models/word2Vec.model'
    model2=train_wordVectors(sentences, embedding_size=128, window=5, min_count=5)
    save_wordVectors(model2,word2vec_path)
    model2=load_wordVectors(word2vec_path)
    print(model2.wv.similarity('沙瑞金', '高育良'))

