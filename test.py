import pickle
import os
import numpy as np
# root = "/Users/liangyunfan/Downloads/EHR_papers_lyf/medical_data/data"
# item_code_index = pickle.load(open(os.path.join(root, "item_code_index.pkl"), "rb"))
# item_code_index = dict([(value, key) for key, value in item_code_index.items()])
# # print("item_code_index:", item_code_index)
# a = item_code_index.keys()
# # print(a)
# root1 = "../real_data/"
# true_data = np.load(os.path.join(root1, "medical_data_effect_test.npy"))
# print(true_data.shape)


# path_d = '/Users/liangyunfan/Downloads/MIMIC3_data/DIAGNOSES_ICD.csv'
import pandas as pd
# data = pd.read_csv(path_d)
# disease = data['ICD9_CODE']
# event_num = data['SEQ_NUM']
# event_num = [i for i in event_num]
# print(max(event_num))
# a = set([i for i in disease])
# a = np.array([i for i in a])
# # print(a.shape)

# path_m = '/Users/liangyunfan/Downloads/EHR_papers_lyf/medical_data/MED_Att_multi/DIAGNOSE.npy'
# data = np.load(path_m)
# print(data.shape)
# x_dim = data.shape[0]
# a=[]
#
# for i in range(x_dim):
#     if sum(data[i])>1000:
#         a.append(i)
#
# print(a)
# print(sum(data[2]))


#################################################################################################################################
# item_code index
import pickle

def gen_itemcode_index():
    ICD_path = '/Users/liangyunfan/Downloads/MIMIC3_data/D_ICD_DIAGNOSES.csv'
    diagnose_path = '/Users/liangyunfan/Downloads/MIMIC3_data/DIAGNOSES_ICD.csv'

    ICD_CODE = pd.read_csv(diagnose_path)
    ICD_CODE = ICD_CODE['ICD9_CODE']
    ICD_CODE = set([i for i in ICD_CODE])
    ICD_CODE = np.array([i for i in ICD_CODE])
    ICD_CODE_dict = {}
    for i,j in enumerate(ICD_CODE):
        ICD_CODE_dict[j] = i
    with open("MIMIC_CODE_INDEX.plk",'wb') as f:
        pickle.dump(ICD_CODE_dict,f)
#################################################################################################################################

import numpy as np
def cal_most_disease():
    path = '/Users/liangyunfan/Downloads/MIMIC3_data/DIAGNOSES_ICD.csv'
    data_d = pd.read_csv(path)
    ICD_code = data_d['ICD9_CODE']
    print(ICD_code.shape[0])
    ICD_set = np.array([i for i in set(ICD_code)])
    cal = np.zeros(ICD_code.shape[0])
    dict = {}
    for i,j in enumerate(ICD_set):
        dict[j] = i
    for i in ICD_code:
        cal[dict.get(i)] = cal[dict.get(i)]+1

    index = np.where(cal == max(cal))[0][0]
    ICD_max = []
    for i in dict.keys():
        if dict.get(i) == index:
            ICD_max.append(i)
    print('\nthe most frequent ICD is: {}    number is: {}'.format(ICD_max,max(cal)))

# cal_most_disease()

# path_onehot = 'DIAGNOSE.npy'
# data = np.load(path_onehot)
# print('shape:  ',data.shape)
# NON=[]
# for i,j in enumerate(data.sum(axis=1)):
#     if j == 0:
#         NON.append(i)
# print(len(NON),data.shape[0])


############################################################################################################################################################
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import *
from sklearn.tree import DecisionTreeClassifier


def gene_diabetes(data):
    import seaborn as sns

    diaarr = np.zeros([1, data.shape[1]])
    list = [25000,
            25001,
            25002,
            25003,
            25010,
            25011,
            25012,
            25013,
            25020,
            25021,
            25022,
            25023,
            25030,
            25031,
            25032,
            25033,
            25040,
            24900,
            24901,
            24910,
            24911,
            24920,
            24921,
            24930,
            24931,
            24940,
            24941,
            24950,
            24951,
            24960,
            24961,
            24970,
            24971,
            24980,
            24981,
            24990,
            24991,
            25041,
            25042,
            25043,
            25050,
            25051,
            25052,
            25053,
            25060,
            25061,
            25062,
            25063,
            25070,
            25071,
            25072,
            25073,
            25080,
            25081,
            25082,
            25083,
            25090,
            25091,
            25092,
            25093,
            2535,
            3572,
            5881,
            64800,
            64801,
            64802,
            64803,
            64804,
            7751]
    print(list)
    list_ = [str(i) for i in list]
    with open('ICDCODEDICR.plk', 'rb') as f:
        ICD_dict = pickle.load(f)
    for ICD in list_:
        if ICD not in ICD_dict.keys():
            continue
        index = ICD_dict[ICD]
        for i in range(data.shape[0]):
            if data[i][index] != 0:
                diaarr = np.concatenate([diaarr, [data[i]]], axis=0)

    diaarr_sel = np.sort(diaarr.sum(axis = 0))[:20]
    datamat =[]

    datamat = pd.DataFrame(datamat)
    data.corr(datamat)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(datamat, vmax=.8, square=True)
    plt.show()

# path_onehot = 'DIAGNOSE.npy'
# data = np.load(path_onehot)
# gene_diabetes(data)

def BR():

    path_onehot = 'DIAGNOSE.npy'
    data = np.load(path_onehot)
    from sklearn import svm
    data = gene_diabetes(data)
    # data = data[:8000]
    train_shape = int(data.shape[0]*0.7)
    clf = BinaryRelevance(DecisionTreeClassifier(max_depth=1))
    clf.fit(data[:train_shape,:6985],data[:train_shape,6985:])
    pre_result = clf.predict(data[train_shape:,:6985])
    Hamming_loss = hamming_loss(pre_result,data[train_shape:,6985:])
    print(Hamming_loss)
    a = pre_result - data[train_shape:,6985:]
    print(a.shape)
    count = 0
    count1=0
    for i in range(492):
        for j in range(2726):
            if a[i,j]==1:
                count+=1
            if a[i,j]==-1:
                count1+=1
    # count = count/(500-train_shape)
    print('aaa   ',count,'   ',count1)
# BR()

# 0.004397322507009267
def select():
    path_onehot = 'DIAGNOSE.npy'
    data = np.load(path_onehot)

    x=data[:,:6985]
    hold = 0
    index = 0
    for i in range(6985):
        if x.sum(axis = 0)[i]>hold:
            hold = x.sum(axis = 0)[i]
            index = i
    print('max is:',hold,'  index is:',index)

import matplotlib.pyplot as plt
def Cancer():
    x = [i*100 for i in range(5)]
    y = [0.3187,0.0063,0.00463,0.00453,0.0044]
    plt.plot(x,y)
    for a,b in zip(x,y):
        plt.text(a,b,b)
    plt.title('Diabetes')
    plt.xlabel('epoch')
    plt.ylabel('Hanming_loss')
    plt.show()
# Cancer()
def paper():
    path_onehot = 'DIAGNOSE.npy'
    data = np.load(path_onehot)
    ICD = data[:, :6985]
    num = ICD.sum(axis = 1)
    num.sort()
    cc = np.zeros(20)
    j=0
    k=0
    for i in num:
        if i<j*3or i==j*3:
            cc[k]+=1
        if i>j*3:
            j+=1
            k+=1
            cc[k] += 1
    xx=[i*3 for i in range(20)]
    # print(cc,xx)
    plt.bar(xx,cc,width=4,color='#4182B4')
    plt.title('ICD calculation')
    plt.xlabel('ICD num')
    plt.ylabel('patient num')
    plt.grid(linestyle='-.')

    plt.axvline(x = 11.9, c="black", ls="--", lw=1)
    # plt.axhline(x = np.mean(np.array(xx)), color="blue")
    plt.show()
# paper()


def paper_():
    path_onehot = 'DIAGNOSE.npy'
    data = np.load(path_onehot)
    print(data.shape)
    drug = data[:, 6985:]

    num = drug.sum(axis = 1)
    average = np.average(num)
    print('...', average)
    num.sort()
    print(max(num))
    cc = np.zeros(20)
    j=0
    k=0
    for i in num:
        if i<j*3 or i==j*3:
            cc[k]+=1
        if i>j*3:
            j+=1
            k+=1
            print(k,j)
            cc[k] += 1
    xx=[i*3 for i in range(20)]
    # print(cc,xx)
    plt.bar(xx,cc,width=4,color='#4182B4')
    plt.title('Drug calculation')
    plt.grid(linestyle='-.')
    plt.axvline(x=12.97, c="black", ls="--", lw=1)
    plt.xlabel('Drug num')
    plt.ylabel('patient num')
    plt.show()


# paper_()

from sklearn.manifold import TSNE
# def distribution():
#     tsne = TSNE(n_components=2,learning_rate=200,n_iter=3000)

# from skmultilearn.problem_transform import ClassifierChain as CC
# def cc():
#     clf =
def kk():
    a = [4500,4700,4997,5011,5300,5210,5953,6062,8500,9249]
    b=  [2.12,1.58,1.37,1.32,1.25,1.19,1.17,1.16,1.03,0.97]
    a.sort(reverse=True)
    for i in range(10):
        a[i] = a[i]/14727/0.3
    epoch = [i*100 for i in range(2,12)]
    plt.subplot(1,2,1)
    plt.plot(epoch,b,marker='*',c='#003366')
    plt.grid(linestyle='-.')
    plt.xlabel('epoch')
    plt.ylabel('average loss')
    plt.title('Weight loss with epoch')
    plt.subplot(1, 2, 2)
    plt.plot(epoch, a, marker='*', c='#003366')
    plt.xlabel('epoch')
    plt.ylabel('average loss')
    plt.title('Weight loss with epoch (diabetes)')
    plt.grid(linestyle='-.')

    plt.show()
# kk()

def aa():
    color = '#4F659A'
#     a=[
#
#         0.43,
#         0.322,
#         0.347,
#         0.21,
#
#     ]
#
#     b=[
#         0.39,
#         0.348,
#         0.345,
#         0.32,
#         ]
#
#     c = [
#         0.37,
#         0.401,
#         0.474,
#         2.89
#     ]
#
#     d = [
#         0.32,
#         0.455,
#         0.490,
#         3.02
#     ]
#     e = [
#         0.38,
#         0.361,
#         0.443,
#         3.85
# ]
#     f = [
#         0.26,
#         0.506,
#         0.532,
#         0.97,
#         ]
    a=[
0.51,
0.301,
0.296,
0.63
]
    b = [
0.49,
0.313,
0.301,
1.32
]
    c = [
0.45,
0.400,
0.357,
4.41
]
    d = [0.40,
0.440,
0.362,
3.53

]
    e = [0.47,
0.372,
0.312,
3.91

]
    f = [0.39,
    0.479,
    0.390,
    1.01
    ]





    x = np.arange(4)


    #
    bar_width = 0.1
    tick_label = ["Hanming_loss", "Jaccard", "F1_micro", "Weight_loss"]
    #
    plt.bar(x, a, bar_width, color = '#0000FF',label = 'BR',  alpha=0.5)
    plt.bar(x+bar_width, b, bar_width,  color ='#008000' ,label = 'CC',alpha=0.5)
    plt.bar(x+2*bar_width, c, bar_width, color ='#FFD700', label = 'RNN_ATT',tick_label = tick_label,alpha=0.5)
    plt.bar(x + 3*bar_width, d, bar_width, color ='#800080', label = 'Med_AR', alpha=0.5)
    plt.bar(x+4*bar_width, e, bar_width, color='#808080', label = 'C2AE',alpha=0.5)
    plt.bar(x+5*bar_width, f, bar_width, color = '#8B0000',label = 'CAAE_SPARSE', alpha=0.5)

    plt.xlabel("Evaluation")
    plt.ylabel("model performance")
    plt.legend(['BR','CC','RNN_ATT','Med_AR','C2AE','CAAE_SPARSE'])

    # # plt.xticks(x + bar_width / 2, tick_label)
    #
    # plt.legend()
    #
    plt.show()
# aa()




ip_convert = lambda x:sum([256**j*int(i) for j,i in enumerate(x.split('.')[::-1])])
def call():
        # 需要手动赋值
        src_ip = ip_convert("192.168.0.31")
        dst_ip = ip_convert("192.168.0.30")

        upper_ptc = 17
        src_port = 10
        dst_port = 20
        seg_len = 10
        app_data = [0x4869]


# 以下自动计算
# IP pseudo-header
# src address + dest address +
# upper layer protocol (padding 8 “0s” to the front) +
# segment length (datagram length – header length)

        src_ip1 = src_ip >> 16
        src_ip2 = src_ip & 0xffff
        dst_ip1 = dst_ip >> 16
        dst_ip2 = dst_ip & 0xffff
        data=[src_ip1, src_ip2, dst_ip1, dst_ip2, upper_ptc, src_port, dst_port, seg_len, seg_len]

        # data = [0xc0da,0x001f,0xc0da,0x001e,0x0011,0x000a,0x0014,0x000a,0x000a]
        data = data + app_data
        print("data:", data)
        """
        def carry_around_add(a, b):
            c = a + b
            return (c & 0xffff) + (c >> 16)
        
        def checksum(msg):
            s = 0
            for i in range(0, len(data)):
                s = carry_around_add(s, data[i])
            return ~s & 0xffff
        
        print("checksum:", hex(checksum(data))[2:])
        """

        # data = [0xc0a8, 0x0166, 0x7b7d, 0x737e, 0x0006, 0x0688, 0x0050, 0x002c, 0x002c, 0x0204, 0x05b4, 0x0103, 0x0303, 0x0101, 0x080a, 0x0101, 0x0402]
        ans = 0
        for i in range(0, len(data)):
            ans = ans + data[i]
        ans = (ans & 0xffff) + (ans >> 16)
        print("checksum hex:", hex(~ans & 0xffff)[2:])
        print("checksum bin:", bin(~ans & 0xffff)[2:])


# call()
def fib():
    for _ in range(3):
        print(1)
fib()