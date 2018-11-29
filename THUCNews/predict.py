#! /usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import numpy as np
import os
from text_cnn import TextCNN
from utils import create_batch_data, create_word2vec, files_processing
import math


def text_predict(files_list, labels_file, models_path, word2vec_path, batch_size):
    '''
    预测...
    :param val_dir:   val数据目录
    :param labels_file:  labels文件目录
    :param models_path:  模型文件
    :param word2vec_path: 词向量模型文件
    :param batch_size: batch size
    :return:
    '''
    max_sentence_length = 300
    embedding_dim = 128
    filter_sizes = [3, 4, 5, 6]
    num_filters = 200  # Number of filters per filter size
    l2_reg_lambda = 0.0  # "L2 regularization lambda (default: 0.0)
    print("Loading data...")
    w2vModel = create_word2vec.load_wordVectors(word2vec_path)

    labels_set = files_processing.read_txt(labels_file)
    labels_nums = len(labels_set)
    sample_num=len(files_list)

    labels_list=[-1]
    labels_list=labels_list*sample_num

    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            cnn = TextCNN(sequence_length = max_sentence_length,
                          num_classes = labels_nums,
                          embedding_size = embedding_dim,
                          filter_sizes = filter_sizes,
                          num_filters = num_filters,
                          l2_reg_lambda = l2_reg_lambda)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, models_path)

            def pred_step(x_batch):
                """
                predictions model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                pred = sess.run([cnn.predictions],feed_dict)
                return pred

            batchNum = int(math.ceil(1.0 * sample_num / batch_size))
            for i in range(batchNum):
                start = i * batch_size
                end = min((i + 1) * batch_size, sample_num)
                batch_files = files_list[start:end]

                # 读取文件内容，字词分割
                batch_content= files_processing.read_files_list_to_segment(batch_files,
                                                                max_sentence_length,
                                                                padding_token='<PAD>')
                # [1]将字词转为索引矩阵,再映射为词向量
                batch_indexMat = create_word2vec.word2indexMat(w2vModel, batch_content, max_sentence_length)
                val_batch_data = create_word2vec.indexMat2vector_lookup(w2vModel, batch_indexMat)

                # [2]直接将字词映射为词向量
                # val_batch_data = create_word2vec.word2vector_lookup(w2vModel,batch_content)

                pred=pred_step(val_batch_data)
                 
                pred=pred[0].tolist()
                pred=files_processing.labels_decoding(pred,labels_set)
                for k,file in enumerate(batch_files):
                    print("{}, pred:{}".format(file,pred[k]))

def batch_predict(val_dir,labels_file,models_path,word2vec_path,batch_size):
    '''
    预测...
    :param val_dir:   val数据目录
    :param labels_file:  labels文件目录
    :param models_path:  模型文件
    :param word2vec_path: 词向量模型文件
    :param batch_size: batch size
    :return:
    '''
    max_sentence_length = 300
    embedding_dim = 128
    filter_sizes = [3, 4, 5, 6]
    num_filters = 200  # Number of filters per filter size
    l2_reg_lambda = 0.0  # "L2 regularization lambda (default: 0.0)
    print("Loading data...")
    w2vModel = create_word2vec.load_wordVectors(word2vec_path)

    labels_set = files_processing.read_txt(labels_file)
    labels_nums = len(labels_set)


    val_file_list = create_batch_data.get_file_list(file_dir=val_dir, postfix='*.npy')
    val_batch = create_batch_data.get_data_batch(val_file_list, labels_nums=labels_nums, batch_size=batch_size,
                                                 shuffle=False, one_hot=True)

    print("val data   info *****************************")
    val_nums = create_word2vec.info_npy(val_file_list)
    print("labels_set info *****************************")
    files_processing.info_labels_set(labels_set)
    # Training
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            cnn = TextCNN(sequence_length = max_sentence_length,
                          num_classes = labels_nums,
                          embedding_size = embedding_dim,
                          filter_sizes = filter_sizes,
                          num_filters = num_filters,
                          l2_reg_lambda = l2_reg_lambda)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, models_path)

            def dev_step(x_batch, y_batch):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                loss, accuracy = sess.run(
                    [cnn.loss, cnn.accuracy],
                    feed_dict)
                return loss, accuracy

            val_losses = []
            val_accs = []
            for k in range(int(val_nums/batch_size)):
            # for k in range(int(10)):
                val_batch_data, val_batch_label = create_batch_data.get_next_batch(val_batch)
                val_batch_data = create_word2vec.indexMat2vector_lookup(w2vModel, val_batch_data)
                val_loss, val_acc=dev_step(val_batch_data, val_batch_label)
                val_losses.append(val_loss)
                val_accs.append(val_acc)
                print("--------Evaluation:step {}, loss {:g}, acc {:g}".format(k, val_loss, val_acc))

            mean_loss = np.array(val_losses, dtype=np.float32).mean()
            mean_acc = np.array(val_accs, dtype=np.float32).mean()
            print("--------Evaluation:step {}, mean loss {:g}, mean acc {:g}".format(k, mean_loss, mean_acc))


def main():
    # Data preprocess
    labels_file = 'data/THUCNews_labels.txt'
    # word2vec_path = 'word2vec/THUCNews_word2vec300.model'
    word2vec_path = "../word2vec/models/THUCNews_word2Vec/THUCNews_word2Vec_128.model"
    models_path='models/checkpoints/model-30000'
    batch_size = 128
    val_dir = './data/val_data'

    batch_predict(val_dir=val_dir,
          labels_file=labels_file,
          models_path=models_path,
          word2vec_path=word2vec_path,
          batch_size=batch_size)

    test_path='/home/ubuntu/project/tfTest/THUCNews/my_test'
    files_list = files_processing.get_files_list(test_path,postfix='*.txt')
    text_predict(files_list, labels_file, models_path, word2vec_path, batch_size)

if __name__=="__main__":
    main()


