import numpy as np
import tensorflow as tf
from pandas.core.frame import DataFrame

def shuff5(Clus,PS,TCP,ACP,NCP):
    a = np.random.random_integers(100, 100000)
    np.random.seed(a)
    np.random.shuffle(Clus)
    np.random.seed(a)
    np.random.shuffle(PS)
    np.random.seed(a)
    np.random.shuffle(TCP)
    np.random.seed(a)
    np.random.shuffle(ACP)
    np.random.seed(a)
    np.random.shuffle(NCP)
    tf.random.set_seed(a)
def shuff2(Final_code,Clus):
    a = np.random.random_integers(100, 100000)
    np.random.seed(a)
    np.random.shuffle(Final_code)
    np.random.seed(a)
    np.random.shuffle(Clus)
    tf.random.set_seed(a)
def shuffALL(Clus,PS,TCP,ACP,NCP,dis1,dis2,dis3,dis4):
    a = np.random.random_integers(100, 100000)
    np.random.seed(a)
    np.random.shuffle(Clus)
    np.random.seed(a)
    np.random.shuffle(PS)
    np.random.seed(a)
    np.random.shuffle(TCP)
    np.random.seed(a)
    np.random.shuffle(ACP)
    np.random.seed(a)
    np.random.shuffle(NCP)
    np.random.seed(a)
    np.random.shuffle(dis1)
    np.random.seed(a)
    np.random.shuffle(dis2)
    np.random.seed(a)
    np.random.shuffle(dis3)
    np.random.seed(a)
    np.random.shuffle(dis4)
def build_loc_code(PS,weight):
    PS_base = []
    PS_temp = []
    for sequence in PS:
        a_num = sequence.count('a')
        c_num = sequence.count('c')
        g_num = sequence.count('g')
        t_num = sequence.count('t')
        a_count = 0
        c_count = 0
        g_count = 0
        t_count = 0

        for loc in sequence:
            if loc == 'a':
                a_count += 1
                a_temp = a_count / a_num
                PS_temp.append(a_temp * weight)
            elif loc == 'c':
                c_count += 1
                c_temp = c_count / c_num
                PS_temp.append(c_temp * weight)
            elif loc == 'g':
                g_count += 1
                g_temp = g_count / g_num
                PS_temp.append(g_temp * weight)
            elif loc == 't':
                t_count += 1
                t_temp = t_count / t_num
                PS_temp.append(t_temp * weight)
            elif loc=='0':
                PS_temp.append(0)
        if PS_temp != []:
            PS_base.append(PS_temp) # 一维是序列长度,二维是每个序列的长度上的值
            PS_temp = []
    PS_DF = DataFrame(PS_base)  # 213*2925
    kmer = np.expand_dims(PS_DF, axis=2)  # 213*2925*1
    return kmer
def OneHot(seq, len_max,weight):
    arrays = np.zeros((0, len_max, 4)) # arrays 0*2925*4
    for i in range(len(seq)):
        cur_seq = seq[i]
        array = np.empty((0, 4)) # array= 0*4
        for j in cur_seq:
            if j == "a":
                array = np.append(array, [[weight, 0, 0, 0]], axis=0)
            if j == "t":
                array = np.append(array, [[0, weight, 0, 0]], axis=0)
            if j == "c":
                array = np.append(array, [[0, 0, weight, 0]], axis=0)
            if j == "g":
                array = np.append(array, [[0, 0, 0, weight]], axis=0)
            if j == "0":
                array = np.append(array, [[0, 0, 0, 0]], axis=0)
        # array:2925*4 ——> array=1*2925*4
        array = array[np.newaxis,:]
        # 最终为213*2925*4
        arrays = np.append(arrays, array, axis=0)
    return arrays
def buildOnlyKmer(kmer):
    init_kmer = kmer
    final_kmer = kmer
    for i in range(4):
        final_kmer = np.concatenate((final_kmer, init_kmer), axis=2)  # (213, 2925, 1->2—>3->4->5)
    return final_kmer
def buildOnlyOneHot(onehot):
    tmp=np.zeros((213,2925,1))
    final_onehot=np.concatenate((onehot, tmp), axis=2)  # (213, 2925, 5)
    return final_onehot
def addDistance(dis1,dis2,dis3,dis4):
    dis5 = [0] * 213
    dis1 = DataFrame(dis1)  # 213*1
    dis1 = np.expand_dims(dis1, axis=2)  # c1:213*1*1

    dis2 = DataFrame(dis2)  # 213*1
    dis2 = np.expand_dims(dis2, axis=2)  # c1:213*1*1

    dis3 = DataFrame(dis3)  # 213*1
    dis3 = np.expand_dims(dis3, axis=2)  # c1:213*1*1

    dis4 = DataFrame(dis4)  # 213*1
    dis4 = np.expand_dims(dis4, axis=2)  # c1:213*1*1

    dis5 = DataFrame(dis5)  # 213*1
    dis5 = np.expand_dims(dis5, axis=2)  # c1:213*1*1

    fianlDis = np.concatenate((dis1, dis2), axis=2)  # (213, 1, 2)
    fianlDis = np.concatenate((fianlDis, dis3), axis=2)  # (213, 1, 3)
    fianlDis = np.concatenate((fianlDis, dis4), axis=2)  # (213, 1, 4)
    fianlDis = np.concatenate((fianlDis, dis5), axis=2)  # (213, 1, 5)

    return fianlDis
def UpDimension(CP):
    c = DataFrame(CP)  # 213*1
    c_1 = np.expand_dims(c, axis=2)  # c1:213*1*1
    c_name = []
    for i in range(1, 6):
        c_name.append('c_' + str(i))
    # c_name:['c_1', 'c_2', 'c_3', 'c_4', 'c_5']

    for i in range(1, len(c_name)):  # [1,5)
        cmd = '%s=np.concatenate((l,c_1),axis=2)' % c_name[i]
        cmd_1 = cmd.replace('l', c_name[i - 1])
        exec(cmd_1)
        '''
        c1:213*1*1
        c2:213*1*2 c1&c1
        c3:213*1*3 c2&c1
        c4:213*1*4 c3&c1
        c5:213*1*5 c4&c1
        '''