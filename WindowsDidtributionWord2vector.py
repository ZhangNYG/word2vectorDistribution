# -*-coding:utf-8 -*-

# ***************Word2vec For Chinese**********************#
# Revised on January 25, 2018 by
# Author: XianjieZhang
# Dalian University of Technology
# email: xj_zh@foxmail.com
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import math
import random

import hdfs
import numpy as np
import time
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# 中文图片乱码解决
from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc")
# 数据路径名字
filename = "135MB.txt"

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('job_name', 'ps', '"ps"or"worker"这个是参数服务器的名字还是计算服务器的名字')
tf.app.flags.DEFINE_string('ps_hosts', '127.0.0.1:22222', '参数服务器列表，192.168.1.160:2222,192.168.1.161:1111')
tf.app.flags.DEFINE_string('worker_hosts', '127.0.0.1:66662', '计算服务器，192.168.1.161:5555,192.168.162:6666')
tf.app.flags.DEFINE_integer('task_id', 0, '当前程序的任务ID，参数服务器和计算服务器都是从0开始')

###############################
# 一些必要设置参数
# 字典中的词语数量
VOCABULARY_SIZE = 2000
# 保存LOG与参数保存时间间隔
SAVE_LOG_TIME = 60  # 秒为单位
# 保存LOG与参数相对路劲
SAVE_LOG_PATH = 'save_log'
# 学习率衰减一次所需全局步数 0.99的衰减率
RATE_DECAY = 50000
# 词向量保存路径
SAVE_NPY = 'save_npy'
# 图片保存路径
SAVE_PIC = 'save_pic'
# 分布式集群机器数量
NUM_COMPUTER = 2
# hadoop中的路径

HADOOP_IP_PORT = "http://192.168.1.160:50070"
HADOOP_PATH = ["/hadoopTest/", "/hadoopTest1/", "/hadoopTest2/"]
###############################
BATCH_SIZE = 128  # 一次训练词的数量
EMBEDDING_SIZE = 128  # Dimension of the embedding vector. 词向量维度
SKIP_WINDOW = 1  # How many words to consider left and right.
NUM_SKIPS = 2  # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
VALID_SIZE = 16  # Random set of words to evaluate similarity on.
VALID_WINDOW = 100  # Only pick dev samples in the head of the distribution.
VALID_EXAMPLES = np.random.choice(VALID_WINDOW, VALID_SIZE, replace=False)
NUM_SAMPLED = 2  # Number of negative examples to sample. NUM_SAMPLED = 64  这个是负样本个数，正样本的个数


# 在这是1(labels的第二个维度)，

###########################################################################
# 数据不能直接读入要循环读取！！待做
def read_data(client,filename):
    with client.read(filename,encoding='utf-8') as f:
        data = []
        counter = 0
        for line in f:
            line = line.strip('\n').strip('').strip('\r')
            if line != "":
                counter += 1
                data_tmp = [word for word in line.split(" ") if word != '']
            data.extend(data_tmp)
            # print(data_tmp)
    return data


############################################################################
# 创建文件夹
def mkdir(path):
    # 去掉路径空格
    path = path.strip()
    # 判断路径是否存在
    isExists = os.path.exists(path)

    if not isExists:
        # 如果不存在就创建目录
        os.makedirs(path)
        print(path + " 路径创建成功")
    else:
        print(path + " 路径已存在")


##############################################################################
# Step 2: Build the dictionary and replace rare words with UNK token.
# 对文件进行编码
def build_dataset(words,dictionary):
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    #reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data


####################################################################################


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(data, batch_size, num_skips, skip_window):
    global global_data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window  # 判断是否正常
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ] 3
    buffer = collections.deque(maxlen=span)  # span=3   deque是一个队列
    for _ in range(span):
        buffer.append(data[global_data_index])
        global_data_index = (global_data_index + 1) % len(data)
    for i in range(batch_size // num_skips):  # 算术运算符" // "来表示整数除法，返回不大于结果的一个最大的整数
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[global_data_index])
        global_data_index = (global_data_index + 1) % len(data)
    # Backtrack a little bit to avoid skipping words in the end of a batch
    global_data_index = (global_data_index + len(data) - span) % len(data)
    return batch, labels


#############################################################################################################

# 全局变量global_data_index来记录当前取到哪了，每次取一个batch后会向后移动，如果超出结尾，则又从头开始。global_data_index = (global_data_index + 1) % len(data)
# skip_window是确定取一个词周边多远的词来训练，比如说skip_window是2，则取这个词的左右各两个词，来作为它的上下文词。后面正式使用的时候取值是1，也就是只看左右各一个词。
# 这里的num_skips我有点疑问，按下面注释是说，How many times to reuse an input to generate a label.，但是我觉得如果确定了skip_window之后，完全可以用
# 这边用了一个双向队列collections.deque，第一次遇见，看代码与list没啥区别，从网上简介来看，双向队列主要在左侧插入弹出的时候效率高，但这里并没有左侧的插入弹出呀，所以是不是应该老实用list会比较好呢？
# num_skips=2*skip_window来确定需要reuse的次数呀，难道还会浪费数据源不成？

# Step 4: Build and train a skip-gram model.
# 下面还有一些细节要处理，以后优化参数

#################################################################################
def plot_with_labels(low_dim_embs, labels, filename):
    filename = filename + '.png'
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom',
                     fontproperties=font)
    plt.savefig(filename)
    plt.cla()
    plt.close('all')


############################################################################

def main(_):
    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    # 定义当前机器是ps还是worker，和它的id
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_id)

    if FLAGS.job_name == 'ps':
        # 参数服务器就运行到这
        server.join()
    # # 读取数据
    # words = read_data(filename)
    # print('Data size', len(words))
    # # 统计数据，建立字典
    # data, count, dictionary, reverse_dictionary = build_dataset(words)
    # # 保存字典
    # f_dict = open('dictionary_data.txt', 'w', encoding='utf-8')
    # f_dict.write(str(reverse_dictionary))
    # f_dict.close()
    # # 保存统计字频
    # f_count = open('count_data.txt', 'w', encoding='utf-8')
    # f_count.write(str(count))
    # f_count.close()
    #
    # del words  # Hint to reduce memory.
    # print('Most common words (+UNK)', count[:5])
    # print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

    # 字典数据和统计数据读入
    # 判断字典路径是否存在
    isExists_dic = os.path.exists('dictionary_data.txt')
    if not isExists_dic:
        # 如果不存在就提醒不存在字典
        print('dictionary_data.txt  ' + " 字典文件不存在！！")
        return " 字典文件不存在！！"
    else:
        f = open('dictionary_data.txt', 'r', encoding='utf-8')
        dictionary_file = f.read()
        reverse_dictionary = eval(dictionary_file)
        dictionary = dict(zip(reverse_dictionary.values(), reverse_dictionary.keys()))
        f.close()
    # 判断统计数据路径是否存在
    isExists_count = os.path.exists('count_data.txt')
    if not isExists_count:

        print('count_data.txt  ' + " 统计文件不存在！！")
        return " 统计文件不存在！！"
    else:
        f = open('count_data.txt', 'r', encoding='utf-8')
        count_file = f.read()
        count = eval(count_file)
        f.close()
    # 判断hadoop数据路径是否存在
    isExists_reverse_path_file_dict = os.path.exists('reverse_path_file_dict.txt')
    if not isExists_reverse_path_file_dict:

        print('reverse_path_file_dict.txt  ' + " hadoop数据文件路径不存在！！")
        return " hadoop路径文件不存在！！"
    else:
        f = open('reverse_path_file_dict.txt', 'r', encoding='utf-8')
        reverse_path_file_dict_file = f.read()
        reverse_path_file_dict = eval(reverse_path_file_dict_file)
        f.close()

    # # data是编完码的数据
    # print('Most common words (+UNK)', count[:5])  # Most common words (+UNK) [['UNK', 169551], ('中国', 32087), ('发展',
    # #  19275), ('人', 11577), ('工作', 10957)]
    # print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])  # Sample data [2379, 36, 0, 2590, 78,
    # #  147, 729, 1419, 0, 3851] ['-', '上海', 'UNK', '实业', '有限公司', '优势', '供应', '机械', 'UNK', '配件']
    # # 全局变量 文件读取 词汇量位置
    # global global_data_index
    # global_data_index = 0
    # batch, labels = generate_batch(data=data, batch_size=8, num_skips=2, skip_window=1)
    # for i in range(8):
    #     print(batch[i], reverse_dictionary[batch[i]],
    #           '->', labels[i, 0], reverse_dictionary[labels[i, 0]])
    # 上面代码只是给出一个例子
    # 下面参数的给出才是正式模型的构建的开始

    # 指定计算服务器设备
    with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_id,
            cluster=cluster)):
        # 在主服务器下创建目录
        if FLAGS.task_id == 0:
            # 创建保存向量目录
            mkdir(SAVE_NPY)
            # 创建保存图片路径
            mkdir(SAVE_PIC)
        # Input data.
        #########################################################################
        # 在默认图中建立变量
        train_inputs = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
        train_labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1])
        valid_dataset = tf.constant(VALID_EXAMPLES, dtype=tf.int32)

        # Ops and variables pinned to the CPU because of missing GPU implementation
        # with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        embeddings = tf.Variable(
            tf.random_uniform([VOCABULARY_SIZE, EMBEDDING_SIZE], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        # tf.nn.embedding_lookup函数的用法主要是选取一个张量里面索引对应的元素。
        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(
            tf.truncated_normal([VOCABULARY_SIZE, EMBEDDING_SIZE],
                                stddev=1.0 / math.sqrt(EMBEDDING_SIZE)))
        nce_biases = tf.Variable(tf.zeros([VOCABULARY_SIZE]))

        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.
        loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=nce_weights,
                           biases=nce_biases,
                           labels=train_labels,
                           inputs=embed,
                           num_sampled=NUM_SAMPLED,
                           num_classes=VOCABULARY_SIZE))

        # 学习率按照指数减少
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(1.0,  # 最初的学习率
                                                   global_step,  # 现在学习步数
                                                   RATE_DECAY,  # 每50000步衰减一次
                                                   0.99,  # 衰减指数
                                                   staircase=True)  # 这个为TRUE的时候可以2000步调整一次

        # Construct the SGD optimizer using a learning rate of 1.0.
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

        # 计算词之间的相似
        # Compute the cosine similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(
            normalized_embeddings, valid_dataset)
        similarity = tf.matmul(
            valid_embeddings, normalized_embeddings, transpose_b=True)

        # Add variable initializer.
        init = tf.initialize_all_variables()

        # Step 5: Begin training.
        # num_steps = 10000001
        ##############################################################

        # 定义新的会话
        # 保存模型
        saver = tf.train.Saver()  # 在这报错
        # 会话
        sv = tf.train.Supervisor(
            is_chief=(FLAGS.task_id == 0),  # 判断是否为参数主服务器
            logdir=SAVE_LOG_PATH,  # log保存地址
            init_op=init,  # 参数初始化
            saver=saver,  # 指定保存模型的saver
            global_step=global_step,  # 全局步数记录
            save_model_secs=SAVE_LOG_TIME  # 指定保存模型的时间间隔--秒
        )
        sess_config = tf.ConfigProto(
            allow_soft_placement=True,  # 软配置 ， 可以自动分配硬件
            log_device_placement=True)  # 显示那个设备执行
        # server.target: To create a tf.Session that connects to this server, return:A string containing a session
        # target for this server.
        session = sv.prepare_or_wait_for_session(server.target, config=sess_config)
        # 对所取文件循环
        one_com_num = len(reverse_path_file_dict) // NUM_COMPUTER
        start_num = FLAGS.task_id  + 1
        current_num = start_num - NUM_COMPUTER
        client = hdfs.Client(HADOOP_IP_PORT, root="/", timeout=500, session=False)
        # 轮数
        circle_num = 1
        # 在一个文件中读取到的词汇位置

        global global_data_index
        global_data_index = 0
        data = []
        # 记录平均损失
        average_loss = 0
        # 步数记录
        step = 0
        while not sv.should_stop():
            #################################################
                # 在这对文件进行读取操作
                # 判断是否读取完成
                # //返回商的整数部分
                # 8个文件4台机器，start_num = 0*（8/4）+1=1
                #                 start_num = 1*（8/4）+1=3
            # aa = global_step.eval(session=session)
            if (global_data_index >= len(data)-BATCH_SIZE) or step == 0:
                global_data_index = 0
                current_num += NUM_COMPUTER
                if current_num <= len(reverse_path_file_dict):
                    current_hdfs_path = reverse_path_file_dict[current_num]
                    words = read_data(client,current_hdfs_path)
                else:
                    current_num = start_num
                    current_hdfs_path = reverse_path_file_dict[current_num]
                    words = read_data(client,current_hdfs_path)
                    circle_num += 1

                # 对word进行编码
                data = build_dataset(words,dictionary)
                del (words)
            # 输入数据，标签数据准备
            batch_inputs, batch_labels = generate_batch(
                data, BATCH_SIZE, NUM_SKIPS, SKIP_WINDOW)
            # 读取完成之后跳转下一个文件

            #################################################

            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            step += 1
            average_loss += loss_val

            #############################################################

            # with tf.Session(graph=graph, config=tf.ConfigProto(
            #         allow_soft_placement=True,
            #         log_device_placement=True)) as session:
            #
            #     # # 恢复前一次训练
            #
            #     saver = tf.train.Saver()
            #     ckpt_state = tf.train.get_checkpoint_state('save/')
            #     if ckpt_state != None:
            #         print("对上次训练进行恢复..........")
            #         print('上次训练模型路径:  ', ckpt_state.model_checkpoint_path)
            #         saver.restore(session, ckpt_state.model_checkpoint_path)
            #     else:
            #         # We must initialize all variables before we use them.
            #         init.run()
            #     print("Initialized")
            #
            #     loss_all = []
            #     average_loss = 0
            #     for step in xrange(num_steps):
            #         batch_inputs, batch_labels = generate_batch(
            #             batch_size, NUM_SKIPS, SKIP_WINDOW)
            #         feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
            #
            #         # We perform one update step by evaluating the optimizer op (including it
            #         # in the list of returned values for session.run()
            #         _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            #         average_loss += loss_val

            # 最后这些保存的过程由一台机器去做就行了，可以设置成每过一段时间保存一次模型，或者图片，由于是异步计算，不会影响整体的进展
            # 这个步数只是这台电脑的步数，应该用图里面的全局步数，或者用时间间隔，时间间隔可以取出来。
            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                # loss_all.append(average_loss)
                print("全局训练步数: ", global_step.eval(session=session))
                print("本机 Average loss at 本机训练步数为: ", step, "    平均损失值: ", average_loss)
                print("本机当前计算文件",current_hdfs_path,"本机词语训练位置: ", global_data_index)  # 词语训练位置
                print("本机当前文件总共词汇量: ", len(data) , "    本机训练第几轮: ",circle_num)
                average_loss = 0
                learn_rate = session.run(learning_rate)
                print("本机当前学习率: ", learn_rate, '\n')

            # 下面判断是否为主服务器，如果是主服务器就用来保存参数与图片。
            if FLAGS.task_id == 0:
                # Note that this is expensive (~20% slowdown if computed every 500 steps)
                if step % 10000 == 0:  # 计算显示一次相似词汇
                    # # 每10000步保存一次模型
                    # if step != 0:
                    #     save_path = saver.save(session, "save/SaveModel.ckpt", global_step=global_step)
                    #     print('保存变量模型save/SaveModel:', tf.train.get_checkpoint_state('save/').model_checkpoint_path)
                    # 计算显示一次相似词汇
                    sim = similarity.eval(session=session)
                    for i in xrange(VALID_SIZE):
                        valid_word = reverse_dictionary[VALID_EXAMPLES[i]]
                        top_k = 8  # number of nearest neighbors
                        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                        log_str = "Nearest to %s:" % valid_word
                        for k in xrange(top_k):
                            close_word = reverse_dictionary[nearest[k]]
                            log_str = "%s   %s ," % (log_str, close_word)
                        print(log_str)

                final_embeddings = normalized_embeddings.eval(session=session)
                # 每1万步保存一次词向量 和 前500词汇图片
                if step % 10000 == 0:
                    if step != 0:
                        np.save(SAVE_NPY + "/vectorForWords.npy", final_embeddings)

                if step % 10000 == 0 and step <= 50000:
                    if step != 0:
                        try:
                            tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
                            plot_only = 500
                            low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
                            labels = [reverse_dictionary[i] for i in xrange(plot_only)]
                            global_step_get = str(global_step.eval(session=session))
                            plot_with_labels(low_dim_embs, labels,
                                             filename=SAVE_PIC + '/globalstep-' + global_step_get + '-' + time.strftime(
                                                 "%Y-%m-%d--%H_%M_%S", time.localtime()))
                        except ImportError:
                            print("save failed OR Please install sklearn, matplotlib, and scipy to visualize "
                                  "embeddings.")

                # 每100万步保存一次全部词汇图片
                if step % 10000 == 0:
                    if step != 0:
                        try:
                            for i_word in xrange(1, VOCABULARY_SIZE - 1000, 500):  # 总字典词汇量减去1000
                                low_dim_embs = tsne.fit_transform(final_embeddings[i_word:i_word + 500, :])
                                labels = [reverse_dictionary[i] for i in xrange(i_word, i_word + 500)]
                                plot_with_labels(low_dim_embs, labels,
                                                 filename=SAVE_PIC + '/words-start-' + str(i_word) + '-picture')
                        except ImportError:
                            print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")
        sv.stop()
        # print("Step5 over")
        # print(type(final_embeddings))

    # numpy.save("filename.npy",a)
    # 利用这种方法，保存文件的后缀名字一定会被置为.npy，这种格式最好只用
    # numpy.load("filename")来读取。
    # Step 6: Visualize the embeddings.
    # 字典也要保存
    # 保存
    # dict_name = {1: {1: 2, 3: 4}, 2: {3: 4, 4: 5}}
    # f = open('temp.txt', 'w')
    # f.write(str(dict_name))
    # f.close()

    # 读取
    # f = open('temp.txt', 'r')
    # a = f.read()
    # dict_name = eval(a)
    # f.close()


if __name__ == "__main__":
    tf.app.run()
