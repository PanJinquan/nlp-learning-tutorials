# -*-coding: utf-8 -*-
"""
    @Project: nlp-learning-tutorials
    @File   : load_data.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-10-29 17:37:57
"""
import time
import pickle
import os
import tensorflow as tf
print(tf.__version__)
# 加载数据
def getFilePathList(rootDir):
    filePath_list = []
    for walk in os.walk(rootDir):
        part_filePath_list = [os.path.join(walk[0], file) for file in walk[2]]
        filePath_list.extend(part_filePath_list)
    return filePath_list

startTime = time.time()
contentListPath_list = getFilePathList('content_list')
content_list = []
for filePath in contentListPath_list:
    with open(filePath, 'rb') as file:
        part_content_list = pickle.load(file)
    content_list.extend(part_content_list)
with open('label_list.pickle', 'rb') as file:
    label_list = pickle.load(file)
used_time = time.time() - startTime
print('used time: %.2f seconds' %used_time)
sample_size = len(content_list)
print('length of content_list，mean sample size: %d' %sample_size)

# 词汇表
from collections import Counter
def getVocabularyList(content_list, vocabulary_size):
    allContent_str = ''.join(content_list)
    counter = Counter(allContent_str)
    vocabulary_list = [k[0] for k in counter.most_common(vocabulary_size)]
    return ['PAD'] + vocabulary_list
startTime = time.time()
vocabulary_list = getVocabularyList(content_list, 10000)
used_time = time.time() - startTime
print('used time: %.2f seconds' %used_time)

# 数据准备
import time
startTime = time.time()
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(content_list, label_list)
train_content_list = train_X
train_label_list = train_y
test_content_list = test_X
test_label_list = test_y
used_time = time.time() - startTime
print('train_test_split used time : %.2f seconds' %used_time)
vocabulary_size = 10000  # 词汇表达小
sequence_length = 600  # 序列长度
embedding_size = 64  # 词向量维度
num_filters = 256  # 卷积核数目
filter_size = 5  # 卷积核尺寸
num_fc_units = 128  # 全连接层神经元
dropout_keep_probability = 0.5  # dropout保留比例
learning_rate = 1e-3  # 学习率
batch_size = 64  # 每批训练大小
word2id_dict = dict([(b, a) for a, b in enumerate(vocabulary_list)])
content2idList = lambda content : [word2id_dict[word] for word in content if word in word2id_dict]
train_idlist_list = [content2idList(content) for content in train_content_list]
used_time = time.time() - startTime
print('content2idList used time : %.2f seconds' %used_time)
import numpy as np
num_classes = np.unique(label_list).shape[0]
import tensorflow.contrib.keras as kr
train_X = kr.preprocessing.sequence.pad_sequences(train_idlist_list, sequence_length)
from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
train_y = labelEncoder.fit_transform(train_label_list)
train_Y = kr.utils.to_categorical(train_y, num_classes)
import tensorflow as tf
tf.reset_default_graph()
X_holder = tf.placeholder(tf.int32, [None, sequence_length])
Y_holder = tf.placeholder(tf.float32, [None, num_classes])
used_time = time.time() - startTime
print('data preparation used time : %.2f seconds' %used_time)

# 搭建神经网络
embedding = tf.get_variable('embedding',
                            [vocabulary_size, embedding_size])
embedding_inputs = tf.nn.embedding_lookup(embedding,
                                          X_holder)
conv = tf.layers.conv1d(embedding_inputs,
                        num_filters,
                        filter_size)
max_pooling = tf.reduce_max(conv,
                            [1])
full_connect = tf.layers.dense(max_pooling,
                               num_fc_units)
full_connect_dropout = tf.contrib.layers.dropout(full_connect,
                                                 keep_prob=dropout_keep_probability)
full_connect_activate = tf.nn.relu(full_connect_dropout)
softmax_before = tf.layers.dense(full_connect_activate,
                                 num_classes)
predict_Y = tf.nn.softmax(softmax_before)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_holder,
                                                           logits=softmax_before)
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)
isCorrect = tf.equal(tf.argmax(Y_holder, 1), tf.argmax(predict_Y, 1))
accuracy = tf.reduce_mean(tf.cast(isCorrect, tf.float32))

# 参数初始化
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)

# 模型训练
test_idlist_list = [content2idList(content) for content in test_content_list]
test_X = kr.preprocessing.sequence.pad_sequences(test_idlist_list, sequence_length)
test_y = labelEncoder.transform(test_label_list)
test_Y = kr.utils.to_categorical(test_y, num_classes)
saver = tf.train.Saver()

import random
for i in range(20000):
    selected_index = random.sample(list(range(len(train_y))), k=batch_size)
    batch_X = train_X[selected_index]
    batch_Y = train_Y[selected_index]
    session.run(train, {X_holder:batch_X, Y_holder:batch_Y})
    step = i + 1
    if step % 500 == 0:
        selected_index = random.sample(list(range(len(test_y))), k=200)
        batch_X = test_X[selected_index]
        batch_Y = test_Y[selected_index]
        loss_value, accuracy_value = session.run([loss, accuracy], {X_holder:batch_X, Y_holder:batch_Y})
        print('step:%d loss:%.4f accuracy:%.4f' %(step, loss_value, accuracy_value))
        # Save the network every 10,000 training iterations
    if (i % 5000 == 0 and i != 0):
        save_path = saver.save(session, "models/pretrained_lstm.ckpt", global_step=i)
        print("saved to %s" % save_path)


# 混淆矩阵
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def predictAll(test_X, batch_size=100):
    predict_value_list = []
    for i in range(0, len(test_X), batch_size):
        selected_X = test_X[i: i + batch_size]
        predict_value = session.run(predict_Y, {X_holder:selected_X})
        predict_value_list.extend(predict_value)
    return np.array(predict_value_list)

Y = predictAll(test_X)
y = np.argmax(Y, axis=1)
predict_label_list = labelEncoder.inverse_transform(y)
pd.DataFrame(confusion_matrix(test_label_list, predict_label_list),
             columns=labelEncoder.classes_,
             index=labelEncoder.classes_ )

# 报告表
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def eval_model(y_true, y_pred, labels):
    # 计算每个分类的Precision, Recall, f1, support
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred)
    # 计算总体的平均Precision, Recall, f1, support
    tot_p = np.average(p, weights=s)
    tot_r = np.average(r, weights=s)
    tot_f1 = np.average(f1, weights=s)
    tot_s = np.sum(s)
    res1 = pd.DataFrame({
        u'Label': labels,
        u'Precision': p,
        u'Recall': r,
        u'F1': f1,
        u'Support': s
    })
    res2 = pd.DataFrame({
        u'Label': ['总体'],
        u'Precision': [tot_p],
        u'Recall': [tot_r],
        u'F1': [tot_f1],
        u'Support': [tot_s]
    })
    res2.index = [999]
    res = pd.concat([res1, res2])
    return res[['Label', 'Precision', 'Recall', 'F1', 'Support']]

eval_model(test_label_list, predict_label_list, labelEncoder.classes_)