# -*-coding: utf-8 -*-
"""
    @Project: LSTM
    @File   : load_data.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-10-21 10:26:54
"""
import numpy as np
import tensorflow as tf

def load_wordVectors(wordsList_path,wordVectors_path):
    '''
    加载单词列表wordsList,以及对应的词向量wordVectors
    :param wordsList_path:
    :param wordVectors_path:
    :return:
    '''
    wordsList = np.load(wordsList_path)
    print('Loaded the word list!')
    wordsList = wordsList.tolist() # originally loaded as numpy array
    wordsList = [word.decode('UTF-8') for word in wordsList] #Encode words as UTF-8
    wordVectors = np.load(wordVectors_path)
    print('Loaded the word vectors!')
    return wordsList,wordVectors

# 我们也可以在词库中搜索单词，比如 “baseball”，然后可以通过访问嵌入矩阵来得到相应的向量，如下：
def get_wordVectors( wordsList, wordVectors,word):
    '''
    获得“单词word”在wordsList的下标，和在wordVectors对应的词向量
    :param wordsList:
    :param wordVectors:
    :param word: type:str
    :return:
    '''
    baseballIndex = wordsList.index(word)
    return baseballIndex,wordVectors[baseballIndex]

def wordVectors_test(wordsList, wordVectors):
    '''
    现在我们有了向量，我们的第一步就是输入一个句子，然后构造它的向量表示。假设我们现在的输入句子是
    “I thought the movie was incredible and inspiring”。为了得到词向量，我们可以使用 TensorFlow 的嵌入函数。
    这个函数有两个参数，一个是嵌入矩阵（在我们的情况下是词向量矩阵），另一个是每个词对应的索引。
    :param wordsList:
    :param wordVectors:
    :return:
    '''
    maxSeqLength = 2  # Maximum length of sentence
    numDimensions = 300  # Dimensions for each word vector
    firstSentence = np.zeros((maxSeqLength), dtype='int32')
    firstSentence[0] = wordsList.index("i")
    firstSentence[1] = wordsList.index("thought")
    # firstSentence[2] = wordsList.index("the")
    # firstSentence[3] = wordsList.index("movie")
    # firstSentence[4] = wordsList.index("was")
    # firstSentence[5] = wordsList.index("incredible")
    # firstSentence[6] = wordsList.index("and")
    # firstSentence[7] = wordsList.index("inspiring")
    # firstSentence[8] and firstSentence[9] are going to be 0

    print('firstSentence.shape:{}'.format(firstSentence.shape))
    # 显示单词的index
    print('firstSentence:{}'.format(firstSentence))  # Shows the row index for each word

    # 显示单词向量，使用下标查询
    print('wordVectors:{}'.format(wordVectors[firstSentence]))
    print('************** embedding_lookup ***************')
    # tensorflow自带了查询词嵌入向量的的方法：embedding_lookup，相当于：wordVectors[firstSentence]
    with tf.Session() as sess:
        # 输出数据是一个 10*50 的词矩阵，其中包括 10 个词，每个词的向量维度是 50。就是去找到这些词对应的向量
        vextor=tf.nn.embedding_lookup(wordVectors, firstSentence).eval()
        print(vextor.shape,vextor)

if __name__=='__main__':
    # 加载单词列表wordsList,以及对应的词向量wordVectors
    wordsList_path='../training_data/wordsList.npy'
    wordVectors_path='../training_data/wordVectors.npy'
    wordsList, wordVectors=load_wordVectors(wordsList_path,wordVectors_path)
    print("单词列表的个数:{}".format(len(wordsList)))
    print("单词向量的维度:{}".format(wordVectors.shape))

    print('***********************get_wordVectors******************************')
    # 获得“单词word”在wordsList的下标，和在wordVectors对应的词向量
    word='baseball'
    index, vect=get_wordVectors(wordsList, wordVectors, word)
    print("word:{},index:{},\n vect:{}".format(word,index, vect))

    print('************************wordVectors_test*****************************')
    wordVectors_test(wordsList, wordVectors)
