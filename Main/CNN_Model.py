import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from Build import shuffALL
from Build import shuff2
from Build import build_loc_code
from Build import OneHot
from Build import buildOnlyKmer
from Build import buildOnlyOneHot
from Build import addDistance
from keras.layers import Conv1D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import MaxPooling1D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from Main.LossHistory_1 import LossHistory
from pandas.core.frame import DataFrame
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

df = pd.read_excel('Data01.xls')
Total_Consensus_PRE = df['Total consensus PRE'].values
Active_Consensus_PRE = df['Number of +consensus PRE'].values
Negative_Consensus_PRE = df['Number of -consensus PRE'].values
Promoter_Sequence = df['Promoter sequence'].values
Cluster = df['cluster'].values
Distance1 = df['Distance1'].values
Distance2 = df['Distance2'].values
Distance3 = df['Distance3'].values
Distance4 = df['Distance4'].values

TCP = Total_Consensus_PRE.tolist()
ACP = Active_Consensus_PRE.tolist()
NCP = Negative_Consensus_PRE.tolist()
PS = Promoter_Sequence.tolist()
Clus = Cluster.tolist()
dis1 = Distance1.tolist()
dis2 = Distance2.tolist()
dis3 = Distance3.tolist()
dis4 = Distance4.tolist()

len_max = 0
for i in range(len(PS)):
    cur_len = len(PS[i])
    if cur_len > len_max:
        len_max = cur_len

for i in range(len(PS)):
    while len(PS[i]) < len_max:
        PS[i] = PS[i] + '0'


shuffALL(Clus, PS, TCP, ACP, NCP, dis1, dis2, dis3, dis4)
print("Shuffle is ok")

kmer = build_loc_code(PS,1)
onlyKmer = buildOnlyKmer(kmer)
print("onlyKmer is ok")

onehot = OneHot(PS, len_max, 1)
onlyOneHot = buildOnlyOneHot(onehot)
print("onlyOneHot is ok")

onehot_kmer = np.concatenate((onehot, kmer), axis=2)
print("onehot_kmer is ok")

# ===============================================================================================

c = DataFrame(TCP)
c_1 = np.expand_dims(c, axis=2)

c_name = []
for i in range(1, 6):
    c_name.append('c_' + str(i))


for i in range(1, len(c_name)):  # [1,5)
    cmd = '%s=np.concatenate((l,c_1),axis=2)' % c_name[i]
    cmd_1 = cmd.replace('l', c_name[i - 1])
    exec(cmd_1)


b = DataFrame(ACP)
b_1 = np.expand_dims(b, axis=2)

b_name = []
for v in range(1, 6):
    b_name.append('b_' + str(v))

for p in range(1, len(b_name)):
    cmd = '%s=np.concatenate((l,b_1),axis=2)' % b_name[p]
    cmd_1 = cmd.replace('l', b_name[p - 1])
    exec(cmd_1)


a = DataFrame(NCP)
a_1 = np.expand_dims(a, axis=2)

a_name = []
for v in range(1, 6):
    a_name.append('a_' + str(v))

for p in range(1, len(a_name)):
    cmd = '%s=np.concatenate((l,a_1),axis=2)' % a_name[p]
    cmd_1 = cmd.replace('l', a_name[p - 1])
    exec(cmd_1)


TCP_onyKmer = np.concatenate((c_5, onlyKmer), axis=1)  # (213, 2926, 5)
TCP_onlyOneHot = np.concatenate((c_5, onlyOneHot), axis=1)  # (213, 2926, 5)
TCP_onehot_kmer = np.concatenate((c_5, onehot_kmer), axis=1)  # (213, 2926, 5)


ACP_TCP_onyKmer = np.concatenate((b_5, TCP_onyKmer), axis=1)  # (213, 2927, 5)
ACP_TCP_onlyOneHot = np.concatenate((b_5, TCP_onlyOneHot), axis=1)  # (213, 2927, 5)
ACP_TCP_onehot_kmer = np.concatenate((b_5, TCP_onehot_kmer), axis=1)  # (213, 2927, 5)


NCP_ACP_TCP_onyKmer = np.concatenate((a_5, ACP_TCP_onyKmer), axis=1)  # (213, 2928, 5)
NCP_ACP_TCP_onlyOneHot = np.concatenate((a_5, ACP_TCP_onlyOneHot), axis=1)  # (213, 2928, 5)
NCP_ACP_TCP_onehot_kmer = np.concatenate((a_5, ACP_TCP_onehot_kmer), axis=1)  # (213, 2928, 5)


Dis = addDistance(dis1,dis2,dis3,dis4)
Dis2 = np.zeros([213,1,5])
NCP_ACP_TCP_onyKmer = np.concatenate((Dis,NCP_ACP_TCP_onyKmer),axis=1)
NCP_ACP_TCP_onlyOneHot = np.concatenate((Dis,NCP_ACP_TCP_onlyOneHot),axis=1)
NCP_ACP_TCP_onehot_kmer = np.concatenate((Dis,NCP_ACP_TCP_onehot_kmer),axis=1)


NCP_ACP_TCP_onyKmer = np.nan_to_num(NCP_ACP_TCP_onyKmer)  # (213, 2928, 5)
NCP_ACP_TCP_onlyOneHot = np.nan_to_num(NCP_ACP_TCP_onlyOneHot)  # (213, 2928, 5)
NCP_ACP_TCP_onehot_kmer = np.nan_to_num(NCP_ACP_TCP_onehot_kmer)  # (213, 2928, 5)


print("Dis_NCP_ACP_TCP_onyKmer is ok")
print("Dis_NCP_ACP_TCP_onlyOneHot is ok")
print("Dis_NCP_ACP_TCP_onehot_kmer is ok")

init_Clus = Clus

Clus = np.concatenate((Clus, init_Clus), axis=0) # 2
Clus = np.concatenate((Clus, init_Clus), axis=0) # 3


Final_code = np.concatenate((NCP_ACP_TCP_onlyOneHot, NCP_ACP_TCP_onehot_kmer), axis=0) # 2
Final_code = np.concatenate((NCP_ACP_TCP_onyKmer, Final_code), axis=0) # 3


shuff2(Final_code, Clus)

z = np.zeros((len(Final_code), 30, 5))  # z:len(Final_code)*30*5
Final_code = np.concatenate((z, Final_code), axis=1)  # (639, 2958, 5)

KFolds = StratifiedKFold(n_splits=10)
fold_counter = 0
result = []
AccList=[]

for train, test in KFolds.split(Final_code, Clus):
    fold_counter += 1
    print(f"fold #{fold_counter}")

    x_train = np.array(Final_code)[train]
    y_train = np.array(Clus)[train]
    x_test = np.array(Final_code)[test]
    y_test = np.array(Clus)[test]


    model = Sequential()


    model.add(Conv1D(filters=8, kernel_size=6, strides=1, padding='same'))  # 卷积层
    model.add(BatchNormalization())  # 批处理层
    model.add(Activation('relu'))  # 激活层
    model.add(MaxPooling1D(pool_size=2))  # 池化层

    model.add(Conv1D(filters=16, kernel_size=12, strides=1, padding='same'))  # 卷积层
    model.add(BatchNormalization())  # 批处理层
    model.add(Activation('relu'))  # 激活层
    model.add(MaxPooling1D(pool_size=4))  # 池化层
    model.add(Dropout(0.2))

    model.add(Flatten())  # 拉直层

    model.add(Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2()))  # 全连接层
    model.add(Dropout(0.2))
    model.add(Dense(5, activation='softmax'))

    history = LossHistory()
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
                    ,metrics=['sparse_categorical_accuracy'])

    model.fit(x_train, y_train, epochs=200,validation_freq=20,callbacks=[history])

    model.summary()
    # model.save("test.h5")
    history.loss_plot('epoch')

    y_pred = model.predict(x_test)
    y_score = [np.argmax(element) for element in y_pred]

    print("Classification Report: \n", classification_report(y_test, y_score))

    acc_count=0
    num=len(y_test)
    for i in range(num):
        if(y_score[i]==y_test[i]):
            acc_count+=1
    print(acc_count/num)
    AccList.append(acc_count/num)

    y_score = pd.DataFrame(y_score, columns=['Pred'])
    y_test = pd.DataFrame(y_test, columns=['Real'])
    Fold_result = pd.concat([y_test, y_score], axis=1)
    result.append(Fold_result)

CNN_result = pd.concat(result, axis=0)
CNN_result.to_csv('CNN_result.csv')


plt.plot(AccList, label='validation accuracy')
plt.title("Validation accuracy of CNN model ")
plt.ylabel('Accuracy')
plt.xlabel('Index')
plt.grid(True)
plt.legend()
plt.show()

print(AccList)