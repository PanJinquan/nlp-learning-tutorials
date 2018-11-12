# -*-coding: utf-8 -*-
"""
    @Project: LSTM
    @File   : train_LSTM.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-10-21 10:57:34
"""
import  tensorflow as tf
from random import randint
import numpy as np
import utils.load_data as load_data
import utils.get_data_batch as get_data_batch
def model_train(wordsList_path,wordVectors_path,matrix_path,model_path,batchSize,numClasses,maxSeqLength,numDimensions,itera = 50000,learning_rate=0.001):
    '''
    :param wordsList_path:
    :param wordVectors_path:
    :param matrix_path:
    :param model_path:
    :param batchSize:
    :param numClasses:
    :param maxSeqLength:
    :param numDimensions:
    :param itera:
    :param learning_rate:
    :return:
    '''
    # 载入测试数据（文本的索引矩阵indeMat）
    tf_indexMat, tf_labels = get_data_batch.load_indexMat(matrix_path, squeeze=True)
    batch_indexMat, batch_labels = get_data_batch.get_batch_images(tf_indexMat, tf_labels,
                                                                   batch_size=batchSize, labels_nums=numClasses,
                                                                   one_hot=True, shuffle=False)
    # 载入词向量(indeMat->wordVectors)
    wordsList, wordVectors = load_data.load_wordVectors(wordsList_path, wordVectors_path)

    # 定义模型的输入数据占位符
    input_labels = tf.placeholder(tf.float32, [batchSize, numClasses])
    input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

    # 将索引矩阵indeMat转换为词向量
    # 调用tf.nn.embedding_lookup() 函数来得到我们的词向量。
    # 该函数最后将返回一个三维向量，第一个维度是批处理大小，第二个维度是句子长度，第三个维度是词向量长度。
    data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]), dtype=tf.float32)
    data = tf.nn.embedding_lookup(wordVectors, input_data)

    # 定义LSTM 网络：
    lstmUnits = 64
    lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)# 设置神经元的个数64
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)# 设置dropout参数，避免一些过拟合
    value, _ = tf.nn.dynamic_rnn(cell=lstmCell, inputs=data, dtype=tf.float32) #展开整个网络，并且构建一整个 RNN 模型

    # dynamic RNN 函数的第一个输出可以被认为是最后的隐藏状态向量。
    # 这个向量将被重新确定维度，然后乘以最后的权重矩阵和一个偏置项来获得最终的输出值。
    weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
    bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
    value = tf.transpose(value, [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, weight) + bias)

    # 计算准确率
    correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(input_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

    # 定义交叉熵损失函数来作为损失值，选择 Adam，并且采用默认的学习率0.001。
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=input_labels))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    saver = tf.train.Saver()
    with tf.Session() as sess:  # 开始一个会话
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(itera+1):
            # Next Batch of reviews
            nextBatch, nextBatchLabels = sess.run([batch_indexMat, batch_labels])
            sess.run(optimizer, {input_data: nextBatch, input_labels: nextBatchLabels})
            if (i % 200 == 0 and i != 0):
                loss_ = sess.run(loss, {input_data: nextBatch, input_labels: nextBatchLabels})
                accuracy_ = sess.run(accuracy, {input_data: nextBatch, input_labels: nextBatchLabels})

                print("iteration {}/{}...".format(i + 1, itera),
                      "loss {}...".format(loss_),
                      "accuracy {}...".format(accuracy_))
                # Save the network every 10,000 training iterations
            if (i % 10000 == 0 and i != 0):
                save_path = saver.save(sess, model_path, global_step=i)
                print("saved to %s" % save_path)
        # 停止所有线程
        coord.request_stop()
        coord.join(threads)

def model_test(wordsList_path,wordVectors_path,matrix_path,models_path,batchSize,numClasses,maxSeqLength,numDimensions):
    '''
    :param wordsList_path:
    :param wordVectors_path:
    :param matrix_path:
    :param batchSize:
    :param numClasses:
    :param maxSeqLength:
    :param numDimensions:
    :return:
    '''
    tf.reset_default_graph() #清除默认图形堆栈并重置全局默认图形，避免与Train的图冲突
    # 载入测试数据
    tf_indexMat, tf_labels=get_data_batch.load_indexMat(matrix_path,squeeze=True)
    batch_indexMat, batch_labels = get_data_batch.get_batch_images(tf_indexMat, tf_labels,
                                                                   batch_size=batchSize,labels_nums=numClasses,
                                                                   one_hot=True, shuffle=False)
    # 载入词向量
    wordsList, wordVectors=load_data.load_wordVectors(wordsList_path,wordVectors_path)

    # 定义模型的输入数据占位符
    input_labels = tf.placeholder(tf.float32, [batchSize, numClasses])
    input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

    # 调用tf.nn.embedding_lookup() 函数来得到我们的词向量。该函数最后将返回一个三维向量，
    # 第一个维度是批处理大小，第二个维度是句子长度，第三个维度是词向量长度。
    data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]), dtype=tf.float32)
    data = tf.nn.embedding_lookup(wordVectors, input_data)

    # 定义LSTM 网络：
    lstmUnits = 64
    lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)# 设置神经元的个数64
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=1.0)# 设置dropout参数，避免一些过拟合
    value, _ = tf.nn.dynamic_rnn(cell=lstmCell, inputs=data, dtype=tf.float32) #展开整个网络，并且构建一整个RNN模型。

    # dynamic RNN 函数的第一个输出可以被认为是最后的隐藏状态向量。
    # 这个向量将被重新确定维度，然后乘以最后的权重矩阵和一个偏置项来获得最终的输出值。
    weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
    bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
    value = tf.transpose(value, [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, weight) + bias)

    # 计算准确率
    correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(input_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

    itera = 10
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # models_path=tf.train.latest_checkpoint('models')
        saver.restore(sess, models_path)
        # 开启测试
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(itera):
            # 在会话中取出一个batch的数据
            nextBatch, nextBatchLabels = sess.run([batch_indexMat, batch_labels])
            accuracy_ = sess.run(accuracy, {input_data: nextBatch, input_labels: nextBatchLabels})
            print('**************************')
            print('batch_images shape:{},batch_labels shape:{}'.format(nextBatch.shape, nextBatchLabels.shape))
            print("accuracy {}...".format(accuracy_))
        # 停止所有线程
        coord.request_stop()
        coord.join(threads)

if __name__=='__main__':
    batchSize = 24      # batch大小
    numClasses = 2      # label个数
    maxSeqLength = 250  # 句子最大长度 Maximum length of sentence
    numDimensions = 300 # 词向量的维度 Dimensions for each word vector
    learning_rate=0.001 # 学习率
    itera = 1       # 迭代次数
    model_path="models/pretrained_lstm.ckpt-40000"
    wordsList_path='./training_data/wordsList.npy'
    wordVectors_path='./training_data/wordVectors.npy'
    train_path = 'training_data/train_indexMat.npy'
    val_path = 'training_data/val_indexMat.npy'
    model_train(wordsList_path, wordVectors_path, train_path,model_path,batchSize,numClasses,maxSeqLength,numDimensions,itera,learning_rate)
    model_test(wordsList_path, wordVectors_path, val_path,model_path,batchSize,numClasses,maxSeqLength,numDimensions)