# -*-coding: utf-8 -*-
"""
    @Project: nlp-learning-tutorials
    @File   : TxtStorage.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-10-27 15:52:54
"""
from numpy import *


def write_txt(content, filename, mode='w'):
    """保存txt数据
    :param content:需要保存的数据,type->list
    :param filename:文件名
    :param mode:读写模式:'w' or 'a'
    :return: void
    """
    with open(filename, mode) as f:
        for line in content:
            str_line = ""
            for col, data in enumerate(line):
                if not col == len(line) - 1:
                    # 以空格作为分隔符
                    str_line = str_line + str(data) + " "
                else:
                    # 每行最后一个数据用换行符“\n”
                    str_line = str_line + str(data) + "\n"
            f.write(str_line)


def read_txt(fileName):
    """读取txt数据函数
    :param filename:文件名
    :return: txt的数据列表
    :rtype: list
    Python中有三个去除头尾字符、空白符的函数，它们依次为:
    strip： 用来去除头尾字符、空白符(包括\n、\r、\t、' '，即：换行、回车、制表符、空格)
    lstrip：用来去除开头字符、空白符(包括\n、\r、\t、' '，即：换行、回车、制表符、空格)
    rstrip：用来去除结尾字符、空白符(包括\n、\r、\t、' '，即：换行、回车、制表符、空格)
    注意：这些函数都只会删除头和尾的字符，中间的不会删除。
    """
    txtData = []
    with open(fileName, 'r') as f:
        lines = f.readlines()
        for line in lines:
            lineData = line.rstrip().split(" ")
            data = []
            for l in lineData:
                if is_int(l):  # isdigit() 方法检测字符串是否只由数字组成,只能判断整数
                    data.append(int(l))
                elif is_float(l):  # 判断是否为小数
                    data.append(float(l))
                else:
                    data.append(l)
            txtData.append(data)
    return txtData


def is_int(str):
    # 判断是否为整数
    try:
        x = int(str)
        return isinstance(x, int)
    except ValueError:
        return False


def is_float(str):
    # 判断是否为整数和小数
    try:
        x = float(str)
        return isinstance(x, float)
    except ValueError:
        return False


def merge_list(data1, data2):
    '''
    将两个list进行合并
    :param data1:
    :param data2:
    :return:返回合并后的list
    '''
    if not len(data1) == len(data2):
        return
    all_data = []
    for d1, d2 in zip(data1, data2):
        all_data.append(d1 + d2)
    return all_data


def split_list(data, split_index=1):
    '''
    将data切分成两部分
    :param data: list
    :param split_index: 切分的位置
    :return:
    '''
    data1 = []
    data2 = []
    for d in data:
        d1 = d[0:split_index]
        d2 = d[split_index:]
        data1.append(d1)
        data2.append(d2)
    return data1, data2


if __name__ == '__main__':
    txt_filename = 'test.txt'
    w_data = [['1.jpg', 'dog', 200, 300, 1.0], ['2.jpg', 'dog', 20, 30, -2]]
    print("w_data=", w_data)
    write_txt(w_data, txt_filename, mode='w')
    r_data = read_txt(txt_filename)
    print('r_data=', r_data)
    data1, data2 = split_list(w_data)
    mer_data = merge_list(data1, data2)
    print('mer_data=', mer_data)



