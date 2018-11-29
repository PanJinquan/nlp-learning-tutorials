# THUCTC数据集
THUCTC(THU Chinese Text Classification)是由清华大学自然语言处理实验室推出的中文文本分类工具包，能够自动高效地实现用户自定义的文本分类语料的训练、评测、分类功能。文本分类通常包括特征选取、特征降维、分类模型学习三个步骤。如何选取合适的文本特征并进行降维，是中文文本分类的挑战性问题。我组根据多年在中文文本分类的研究经验，在THUCTC中选取二字串bigram作为特征单元，特征降维方法为Chi-square，权重计算方法为tfidf，分类模型使用的是LibSVM或LibLinear。THUCTC对于开放领域的长文本具有良好的普适性，不依赖于任何中文分词工具的性能，具有准确率高、测试速度快的优点。</br>


## 训练和测试方法：
   下载提供的word2vec模型以及词向量处理好的THUCNews数据，就可以训练和测试了：
- 1.训练：train_val.py
- 2.测试：predict.py
- 3.预训练模型，准确率：93.9%

## 资源下载：

- 1.THUCTC官方数据集，链接: http://thuctc.thunlp.org/message
- 2.THUCTC百度网盘，链接: https://pan.baidu.com/s/1DT5xY9m2yfu1YGaGxpWiBQ 提取码: bbpe
- 3.已经训练好的word2vec模型：链接: https://pan.baidu.com/s/1n4ZgiF0gbY0zsK0706wZiw 提取码: mtrj 
- 4.使用词向量处理的THUCNews数据：链接: https://pan.baidu.com/s/12Hdf36QafQ3y6KgV_vLTsw 提取码: m9dx 

