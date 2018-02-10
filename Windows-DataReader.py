# -*-coding:utf-8 -*-


# ***************Reader For Chinese**********************#
# Revised on January 25, 2018 by
# Author: XianjieZhang
# Dalian University of Technology
# email: xj_zh@foxmail.com
###########################################################################
# 数据不能直接读入要循环读取！！待做
from six.moves import xrange
import hdfs
import collections
import WindowsDidtributionWord2vector

##############################
# 参数设置
VOCABULARY_SIZE = WindowsDidtributionWord2vector.VOCABULARY_SIZE
# hadoop中的路径
HADOOP_IP_PORT = WindowsDidtributionWord2vector.HADOOP_IP_PORT
HADOOP_PATH = WindowsDidtributionWord2vector.HADOOP_PATH


def read_data(client,filename):
    with client.read(filename,encoding='utf-8') as f:
        data = []
        counter = 0
        data_settmp = set()
        for line in f:
            if line not in data_settmp:
                data_settmp.add(line)
                line = line.strip('\n').strip('').strip('\r')
                data_tmp = []
                if line != "":
                    counter += 1
                    data_tmp = [word for word in line.split(" ") if word != '']
                data.extend(data_tmp)
                # print(data_tmp)
        print('counter: ',counter) #9829
        print('data-words: ', len(data))
    return data


############################################################################
##############################################################################
# Step 2: Build the dictionary and replace rare words with UNK token.
# 建立数据字典
def build_dic(collectionsCounter):
    count = [['UNK', -1]]
    count.extend(collectionsCounter.most_common(VOCABULARY_SIZE - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    unk_count = 0
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return count, dictionary, reverse_dictionary


####################################################################################
# 读取数据
if __name__ == '__main__':

    client = hdfs.Client(HADOOP_IP_PORT, root="/", timeout=500, session=False)
    # 存储文件路径名
    path_file_dict = dict()
    collectionsCounter = collections.Counter()
    for file_path in HADOOP_PATH:  # 三种数据集
        fileList = client.list(file_path)
        for file_loop in fileList:  # 每个数据集中有一批数据
            # 对路径进行储存
            path_file_dict[file_path + file_loop] = len(path_file_dict) + 1
            # 产生读取每个文件
            words = read_data(client, file_path + file_loop)
            print("文件路径名称： ", file_path + file_loop, '    Data size: ', len(words))
            count = [['UNK', -1]]
            collectionsCounter = sum((collectionsCounter,collections.Counter(words)),collections.Counter())
            print('collectionsCounter: ',len(collectionsCounter))
    count, dictionary, reverse_dictionary = build_dic(collectionsCounter)
    # 保存字典
    f_dict = open('dictionary_data.txt', 'w', encoding='utf-8')
    f_dict.write(str(reverse_dictionary))
    f_dict.close()
    # 保存统计字频
    f_count = open('count_data.txt', 'w', encoding='utf-8')
    f_count.write(str(count))
    f_count.close()
    # 存储的文件路径名
    reverse_path_file_dict = dict(zip(path_file_dict.values(), path_file_dict.keys()))
    print(reverse_path_file_dict,len(reverse_path_file_dict))
    # 保存hadoop中所有文件路径键值对
    f_count = open('reverse_path_file_dict.txt', 'w', encoding='utf-8')
    f_count.write(str(reverse_path_file_dict))
    f_count.close()



    # aaa = collectionsCounter

    # 保存字典与统计数据



    # for file_loop in fileList:
    #     # 产生读取每个文件
    #     words = read_data(client, HADOOP_PATH + file_loop)
    #     print("文件路径名称： ",HADOOP_PATH+file_loop,'    Data size: ', len(words))
    #     # loop 统计全部字频
    #
    #     # 统计数据，建立字典
    #     data, count, dictionary, reverse_dictionary = build_dataset(words)
    #     # 保存字典
    #     f_dict = open('dictionary_data.txt', 'w', encoding='utf-8')
    #     f_dict.write(str(reverse_dictionary))
    #     f_dict.close()
    #     # 保存统计字频
    #     f_count = open('count_data.txt', 'w', encoding='utf-8')
    #     f_count.write(str(count))
    #     f_count.close()
    #
    #     del words  # Hint to reduce memory.
    #     print('Most common words (+UNK)', count[:5])
    #     print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
