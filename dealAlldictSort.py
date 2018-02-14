# -*-coding:utf-8 -*-

# 判断字典路径是否存在
import operator
import os
from operator import itemgetter

isExists_dic = os.path.exists('dict_all_counter.txt')
if not isExists_dic:
    # 如果不存在就提醒不存在字典
    print('42上面遍历所有文件保存所有词语词频/dict_all_counter.txt  ' + " 字典文件不存在！！")
    # return " 字典文件不存在！！"
else:
    f = open('dict_all_counter.txt', 'r', encoding='utf-8')
    dictionary_file = f.read()
    reverse_dictionary = eval(dictionary_file)
    # dictionary = dict(zip(reverse_dictionary.values(), reverse_dictionary.keys()))
    f.close()
    # list_all_counter = list(reverse_dictionary)
    list_all_counter = list(map(lambda x, y: (x, y), reverse_dictionary.keys(), reverse_dictionary.values()))
    print(list_all_counter[:5])
    print(len(list_all_counter))
    print(type(list_all_counter))
    print(type(list_all_counter[1]))
    # list_all_counter.sort(key=operator.itemgetter(1,0))
    list_all_counter = sorted(list_all_counter, key=itemgetter(0, 1), reverse=True)
    dic_from_all_list = dict()
    for word, _ in list_all_counter:
        dic_from_all_list[word] = len(dic_from_all_list)+1

    f_list = open('list_all_counter.txt', 'w', encoding='utf-8')
    f_list.write(str(list_all_counter))
    f_list.close()

    f_dic_from_list = open('dic_from_all_list.txt', 'w', encoding='utf-8')
    f_dic_from_list.write(str(dic_from_all_list))
    f_dic_from_list.close()
