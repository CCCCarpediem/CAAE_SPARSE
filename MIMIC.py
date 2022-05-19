import pandas as pd
import numpy as np
import os
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import learn
from sklearn.metrics import *
from tensorflow.contrib.layers import l2_regularizer


ICD_path = '/Users/liangyunfan/Downloads/MIMIC3_data/D_ICD_DIAGNOSES.csv'
diagnose_path = '/Users/liangyunfan/Desktop/DIAGNOSES_ICD_filter.csv'
prescription_path = '/Users/liangyunfan/Downloads/PRESCRIPTIONS_1.csv'

ICD_CODE = pd.read_csv(diagnose_path)
ICD_CODE = ICD_CODE['ICD9_CODE']
ICD_CODE = set([i for i in ICD_CODE])
ICD_CODE = np.array([i for i in ICD_CODE])
ICD_CODE_dict = {}
for i,j in enumerate(ICD_CODE):
    ICD_CODE_dict[j] = i

import pickle as plk
with open('ICDCODEDICR.plk','wb') as f:
 plk.dump(ICD_CODE_dict,f)


ICD_num = 6985

def gen_onehot():

    diagnose_data = pd.read_csv(diagnose_path)
    prescription_data = pd.read_csv(prescription_path)
    HADM_ID_p = prescription_data['HADM_ID']
    HADM_ID_p = [i for i in HADM_ID_p]
    print(np.array([i for i in set(HADM_ID_p)]).shape)
    Medicine = prescription_data['DRUG']
    Medicine_set = [i for i in set(Medicine)]
    Med_dict = {}
    for i,j in enumerate(Medicine_set):
        Med_dict[j] = i

    Med_num = np. array(Medicine_set).shape[0]
    print('number of medicine is: {}'.format(Med_num))

    Med_HA_dict = {}
    hold = 0

    for i in range(np.array(HADM_ID_p).shape[0]):
       if hold == 0 :
            add = np.zeros([1,Med_num])
            add[0,Med_dict.get(Medicine[i])] = 1

       if HADM_ID_p[i] == hold:
            add[0,Med_dict.get(Medicine[i])] = 1

       if HADM_ID_p[i] != hold and hold != 0:
            Med_HA_dict[HADM_ID_p[i - 1]] = add
            add = np.zeros([1, Med_num])
            add[0, Med_dict.get(Medicine[i])] = 1
       hold = HADM_ID_p[i]

    print(np.array([i for i in Med_HA_dict.keys()]).shape)
    # print(Med_HA_dict)




    # diagnose_data.sort_values('SUBJECT_ID',inplace=True,ascending=True)
    SUBJECT_ID = diagnose_data['SUBJECT_ID']
    SUBJECT_ICD = diagnose_data['ICD9_CODE']
    HADM_ID = diagnose_data['HADM_ID']
    HADM_ID = [int(i)for i in HADM_ID]
    HADM_ID_set =  [i for i in set(HADM_ID)]
    print(np.array(HADM_ID_set).shape)

    PATIENT_NUM = np.array([i for i in set(SUBJECT_ID)]).shape[0]
    print('the number of patient is: {}'.format(PATIENT_NUM))
    print('the ICD occured is: {}'.format(ICD_num))

    DIAGNOSE = np.zeros([1, ICD_num + Med_num])
    hold = 0

    for x in range(SUBJECT_ID.shape[0]):
        if HADM_ID[x] in Med_HA_dict.keys():
            if hold != HADM_ID[x] and hold == 0:
                add = np.zeros([1,ICD_num])
                add[0,ICD_CODE_dict.get(SUBJECT_ICD[x])] = 1

            if hold != HADM_ID[x] and hold != 0:

                add = np.concatenate((add, Med_HA_dict.get(HADM_ID[x])), axis=1)
                DIAGNOSE = np.concatenate((DIAGNOSE, add), axis=0)
                add = np.zeros([1,ICD_num])
                add[0,ICD_CODE_dict.get(SUBJECT_ICD[x])] = 1

            if hold == HADM_ID[x]:
                add[0,ICD_CODE_dict.get(SUBJECT_ICD[x])] = 1

            hold = HADM_ID[x]
     
        # if x == 200000:
        #     break
        if x%1000==0:
            print(x,'\n')

    DIAGNOSE = DIAGNOSE[1:]
    print('one hot data is:{}'.format(DIAGNOSE.shape))
    print('test:  ',DIAGNOSE.sum(axis = 1))

    np.save('DIAGNOSE.npy',DIAGNOSE)


def Word2Vec_train():
    ICD_word_vec = [x for x in ICD_CODE]
    ICD_word_vec = [y for y in map(lambda x:str(x), ICD_word_vec)]
    ICD_word_vec = [z+'\n' for z in ICD_word_vec]
    with open('MIMIC_ICD.txt','w')as f:
        f.writelines(ICD_word_vec)

    model = Word2Vec(LineSentence('MIMIC_ICD.txt'), size=512, window=5, min_count=1, workers=4, iter=10000)
    # model.save("MIMIC_ICD_512.model")
    model.wv.save_word2vec_format("MIMIC_Word2Vec.vector",binary=False)

# gen_onehot()
# Word2Vec_train()


