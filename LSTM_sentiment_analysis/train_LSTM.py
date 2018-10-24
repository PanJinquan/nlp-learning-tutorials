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
import load_data
from os import listdir
from os.path import isfile, join
maxSeqLength = 10 #Maximum length of sentence
numDimensions = 300 #Dimensions for each word vector
batchSize = 24
lstmUnits = 64
numClasses = 2
maxSeqLength = 250

# 辅助函数
def getTrainBatch(ids):
    '''
    pos nums:12500,neg nums:12500
    :return:
    '''
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        if (i % 2 == 0):#正样本
            num = randint(1, 11499)
            labels.append([1, 0])
        else:
            num = randint(13499, 24999)
            labels.append([0, 1])
        arr[i] = ids[num - 1:num]
    return arr, labels


def getTestBatch(ids):
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        num = randint(11499, 13499)
        if (num <= 12499):
            labels.append([1, 0])
        else:
            labels.append([0, 1])
        arr[i] = ids[num - 1:num]
    return arr, labels


# # RNN Model
# 现在，我们可以开始构建我们的 TensorFlow 图模型。首先，我们需要去定义一些超参数，比如批处理大小，LSTM的单元个数，分类类别和训练次数。
# 与大多数 TensorFlow 图一样，现在我们需要指定两个占位符，一个用于数据输入，另一个用于标签数据。对于占位符，最重要的一点就是确定好维度。
# 标签占位符代表一组值，每一个值都为 [1,0] 或者 [0,1]，这个取决于数据是正向的还是负向的。输入占位符，是一个整数化的索引数组。
def train(wordsList_path,wordVectors_path,train_matrix_path):
    ids = np.load(train_matrix_path)
    wordsList, wordVectors=load_data.load_wordVectors(wordsList_path,wordVectors_path)
    tf.reset_default_graph()
    labels = tf.placeholder(tf.float32, [batchSize, numClasses])
    input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

    # 一旦，我们设置了我们的输入数据占位符，我们可以调用
    # tf.nn.embedding_lookup() 函数来得到我们的词向量。该函数最后将返回一个三维向量，第一个维度是批处理大小，
    # 第二个维度是句子长度，第三个维度是词向量长度。更清晰的表达，如下图所示：

    data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]), dtype=tf.float32)
    data = tf.nn.embedding_lookup(wordVectors, input_data)

    # 现在我们已经得到了我们想要的数据形式，那么揭晓了我们看看如何才能将这种数据形式输入到我们的 LSTM 网络中。
    # 首先，我们使用 tf.nn.rnn_cell.BasicLSTMCell 函数，这个函数输入的参数是一个整数，表示需要几个 LSTM 单元。
    # 这是我们设置的一个超参数，我们需要对这个数值进行调试从而来找到最优的解。
    # 然后，我们会设置一个 dropout 参数，以此来避免一些过拟合。
    # 最后，我们将 LSTM cell 和三维的数据输入到 tf.nn.dynamic_rnn ，这个函数的功能是展开整个网络，并且构建一整个 RNN 模型。
    lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
    value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

    # 堆栈 LSTM 网络是一个比较好的网络架构。也就是前一个LSTM 隐藏层的输出是下一个LSTM的输入。
    # 堆栈LSTM可以帮助模型记住更多的上下文信息，但是带来的弊端是训练参数会增加很多，
    # 模型的训练时间会很长，过拟合的几率也会增加。
    # dynamic RNN 函数的第一个输出可以被认为是最后的隐藏状态向量。这个向量将被重新确定维度，然后乘以最后的权重矩阵和一个偏置项来获得最终的输出值。
    weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
    bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
    value = tf.transpose(value, [1, 0, 2])
    # 取最终的结果值
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, weight) + bias)

    # 接下来，我们需要定义正确的预测函数和正确率评估参数。正确的预测形式是查看最后输出的0-1向量是否和标记的0-1向量相同。
    correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

    # 之后，我们使用一个标准的交叉熵损失函数来作为损失值。对于优化器，我们选择 Adam，并且采用默认的学习率。
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    # # 超参数调整
    # 选择合适的超参数来训练你的神经网络是至关重要的。你会发现你的训练损失值与你选择的优化器（Adam，Adadelta，SGD，等等），
    # 学习率和网络架构都有很大的关系。特别是在RNN和LSTM中，单元数量和词向量的大小都是重要因素。
    #
    # * 学习率：RNN最难的一点就是它的训练非常困难，因为时间步骤很长。那么，学习率就变得非常重要了。
    # 如果我们将学习率设置的很大，那么学习曲线就会波动性很大，如果我们将学习率设置的很小，那么训练过程就会非常缓慢。
    # 根据经验，将学习率默认设置为 0.001 是一个比较好的开始。如果训练的非常缓慢，那么你可以适当的增大这个值，如果训练过程非常的不稳定，
    # 那么你可以适当的减小这个值。
    #
    # * 优化器：这个在研究中没有一个一致的选择，但是 Adam 优化器被广泛的使用。
    # * LSTM单元的数量：这个值很大程度上取决于输入文本的平均长度。而更多的单元数量可以帮助模型存储更多的文本信息，当然模型的训练时间就会增加很多，并且计算成本会非常昂贵。
    # * 词向量维度：词向量的维度一般我们设置为50到300。维度越多意味着可以存储更多的单词信息，但是你需要付出的是更昂贵的计算成本。

    # # 训练
    # 训练过程的基本思路是，我们首先先定义一个 TensorFlow 会话。然后，我们加载一批评论和对应的标签。
    # 接下来，我们调用会话的 run 函数。这个函数有两个参数，第一个参数被称为 fetches 参数，这个参数定义了我们感兴趣的值。
    # 我们希望通过我们的优化器来最小化损失函数。第二个参数被称为 feed_dict 参数。这个数据结构就是我们提供给我们的占位符。
    # 我们需要将一个批处理的评论和标签输入模型，然后不断对这一组训练数据进行循环训练。
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    iterations = 50000
    for i in range(iterations):
        # Next Batch of reviews
        nextBatch, nextBatchLabels = getTrainBatch(ids);
        sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})

        if (i % 1000 == 0 and i != 0):
            loss_ = sess.run(loss, {input_data: nextBatch, labels: nextBatchLabels})
            accuracy_ = sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})

            print("iteration {}/{}...".format(i + 1, iterations),
                  "loss {}...".format(loss_),
                  "accuracy {}...".format(accuracy_))
            # Save the network every 10,000 training iterations
        if (i % 10000 == 0 and i != 0):
            save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
            print("saved to %s" % save_path)

    # 查看上面的训练曲线，我们发现这个模型的训练结果还是不错的。损失值在稳定的下降，正确率也不断的在接近 100% 。
    # 然而，当分析训练曲线的时候，我们应该注意到我们的模型可能在训练集上面已经过拟合了。过拟合是机器学习中一个非常常见的问题，
    # 表示模型在训练集上面拟合的太好了，但是在测试集上面的泛化能力就会差很多。也就是说，如果你在训练集上面取得了损失值是 0 的模型，
    # 但是这个结果也不一定是最好的结果。当我们训练 LSTM 的时候，提前终止是一种常见的防止过拟合的方法。基本思路是，我们在训练集上面进行模型训练，
    # 同事不断的在测试集上面测量它的性能。一旦测试误差停止下降了，或者误差开始增大了，那么我们就需要停止训练了。
    # 因为这个迹象表明，我们网络的性能开始退化了。
    #
    # 导入一个预训练的模型需要使用 TensorFlow 的另一个会话函数，称为 Server ，然后利用这个会话函数来调用 restore 函数。
    # 这个函数包括两个参数，一个表示当前的会话，另一个表示保存的模型。
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('models'))
    # 然后，从我们的测试集中导入一些电影评论。请注意，这些评论是模型从来没有看见过的。
    iterations = 10
    for i in range(iterations):
        nextBatch, nextBatchLabels = getTestBatch();
        print("Accuracy for this batch:", (sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})) * 100)

def test(wordsList_path,wordVectors_path,train_matrix_path):
    ids = np.load(train_matrix_path)
    wordsList, wordVectors=load_data.load_wordVectors(wordsList_path,wordVectors_path)
    tf.reset_default_graph()
    labels = tf.placeholder(tf.float32, [batchSize, numClasses])
    input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

    # 一旦，我们设置了我们的输入数据占位符，我们可以调用
    # tf.nn.embedding_lookup() 函数来得到我们的词向量。该函数最后将返回一个三维向量，第一个维度是批处理大小，
    # 第二个维度是句子长度，第三个维度是词向量长度。更清晰的表达，如下图所示：

    data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]), dtype=tf.float32)
    data = tf.nn.embedding_lookup(wordVectors, input_data)

    # 现在我们已经得到了我们想要的数据形式，那么揭晓了我们看看如何才能将这种数据形式输入到我们的 LSTM 网络中。
    # 首先，我们使用 tf.nn.rnn_cell.BasicLSTMCell 函数，这个函数输入的参数是一个整数，表示需要几个 LSTM 单元。
    # 这是我们设置的一个超参数，我们需要对这个数值进行调试从而来找到最优的解。
    # 然后，我们会设置一个 dropout 参数，以此来避免一些过拟合。
    # 最后，我们将 LSTM cell 和三维的数据输入到 tf.nn.dynamic_rnn ，这个函数的功能是展开整个网络，并且构建一整个 RNN 模型。
    lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
    value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

    # 堆栈 LSTM 网络是一个比较好的网络架构。也就是前一个LSTM 隐藏层的输出是下一个LSTM的输入。
    # 堆栈LSTM可以帮助模型记住更多的上下文信息，但是带来的弊端是训练参数会增加很多，
    # 模型的训练时间会很长，过拟合的几率也会增加。
    # dynamic RNN 函数的第一个输出可以被认为是最后的隐藏状态向量。这个向量将被重新确定维度，然后乘以最后的权重矩阵和一个偏置项来获得最终的输出值。
    weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
    bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
    value = tf.transpose(value, [1, 0, 2])
    # 取最终的结果值
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, weight) + bias)

    # 接下来，我们需要定义正确的预测函数和正确率评估参数。正确的预测形式是查看最后输出的0-1向量是否和标记的0-1向量相同。
    correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

    # 导入一个预训练的模型需要使用 TensorFlow 的另一个会话函数，称为 Server ，然后利用这个会话函数来调用 restore 函数。
    # 这个函数包括两个参数，一个表示当前的会话，另一个表示保存的模型。
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('models'))
    # 然后，从我们的测试集中导入一些电影评论。请注意，这些评论是模型从来没有看见过的。
    iterations = 10
    for i in range(iterations):
        nextBatch, nextBatchLabels = getTestBatch(ids);
        print("Accuracy for this batch:", (sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})) * 100)

if __name__=='__main__':
    wordsList_path='./training_data/wordsList.npy'
    wordVectors_path='./training_data/wordVectors.npy'
    # 训练数据，已经将文本数据转为词向量，并保存在idsMatrix.npy中
    train_matrix_path='./training_data/idsMatrix.npy'
    train(wordsList_path,wordVectors_path,train_matrix_path)