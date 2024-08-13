#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 14:42:08 2023

@author: zeeshan
"""

import optuna
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix, roc_auc_score, precision_recall_curve, auc
import numpy as np
from numpy import mean
from collections import Counter
import re, os, sys
import random
import pandas as pd
import lightgbm as lgb
import pickle
from xgboost import Booster

# Read and preprocess the data
data_path = '/home/zeeshan/m5C_Mobeen/datasets/ind_danio.fasta'

def read_nucleotide_sequences(file):
    
    if os.path.exists(file) == False:
        print('Error: file %s does not exist.' % file)
        sys.exit(1)
    with open(file) as f:
        records = f.read()
    if re.search('>', records) == None:
        print('Error: the input file %s seems not in FASTA format!' % file)
        sys.exit(1)
    records = records.split('>')[1:]

    fasta_sequences = []
    for fasta in records:
        array = fasta.split('\n')
        header, sequence = array[0].split()[0], re.sub('[^ACGTU-]', '-', ''.join(array[1:]).upper())
        header_array = header.split('|')
        name = header_array[0]
        label = header_array[1] if len(header_array) >= 2 else '0'
        label_train = header_array[2] if len(header_array) >= 3 else 'training'
        sequence = re.sub('U', 'T', sequence)
        fasta_sequences.append([name, sequence, label, label_train])
    return fasta_sequences

# Define encoding functions

from Bio import SeqIO
def dataProcessing(path,fileformat):
    all_seq_data = []

    for record in SeqIO.parse(path,fileformat):
        sequences = record.seq # All sequences in dataset

        seq_data=[]
       
        for i in range(len(sequences)):
            if sequences[i] == 'A':
                seq_data.append([1,0,0,0])
            if sequences[i] == 'T':
                seq_data.append([0,1,0,0])
            if sequences[i] == 'U':
                seq_data.append([0,1,0,0])                
            if sequences[i] == 'C':
                seq_data.append([0,0,1,0])
            if sequences[i] == 'G':
                seq_data.append([0,0,0,1])
            if sequences[i] == 'N':
                seq_data.append([0,0,0,0])
                
        all_seq_data.append(seq_data)    
        
    all_seq_data = np.array(all_seq_data);
    
    return all_seq_data
    

def TNC(fastas, **kw):
    AA = 'ACGT'
    encodings = []
    triPeptides = [aa1 + aa2 + aa3 for aa1 in AA for aa2 in AA for aa3 in AA]
    header = ['#', 'label'] + triPeptides
    encodings.append(header)

    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i

    for i in fastas:
        name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
        code = [name, label]
        tmpCode = [0] * 64
        for j in range(len(sequence) - 3 + 1):
            tmpCode[AADict[sequence[j]] * 16 + AADict[sequence[j+1]]*4 + AADict[sequence[j+2]]] = tmpCode[AADict[sequence[j]] * 16 + AADict[sequence[j+1]]*4 + AADict[sequence[j+2]]] +1
        if sum(tmpCode) != 0:
            tmpCode = [i/sum(tmpCode) for i in tmpCode]
        code = code + tmpCode
        encodings.append(code)
    return encodings


def CKSNAP(fastas, gap, **kw):
  
    kw = {'order': 'ACGT'}
    AA = kw['order'] if kw['order'] != None else 'ACGT'
    encodings = []
    aaPairs = []
    for aa1 in AA:
        for aa2 in AA:
            aaPairs.append(aa1 + aa2)

    header = ['#', 'label']
    for g in range(gap + 1):
        for aa in aaPairs:
            header.append(aa + '.gap' + str(g))
    encodings.append(header)

    for i in fastas:
        name, sequence, label = i[0], i[1], i[2]
        code = [name, label]
        for g in range(gap + 1):
            myDict = {}
            for pair in aaPairs:
                myDict[pair] = 0
            sum = 0
            for index1 in range(len(sequence)):
                index2 = index1 + g + 1
                if index1 < len(sequence) and index2 < len(sequence) and sequence[index1] in AA and sequence[
                    index2] in AA:
                    myDict[sequence[index1] + sequence[index2]] = myDict[sequence[index1] + sequence[index2]] + 1
                    sum = sum + 1
            for pair in aaPairs:
                code.append(myDict[pair] / sum)
        encodings.append(code)
    return encodings


# Perform CKSNAP encoding
file=read_nucleotide_sequences(data_path) #ENAC encoded
cks = CKSNAP(file,gap=5) #gap=5 default
cc=np.array(cks)
data_only1 = cc[1:,2:]
# data_only1 = cc[1:201,2:]
data_cksnap = data_only1.astype(np.float)

# Perform One-hot encoding
data_io = dataProcessing(data_path,"fasta") #path,fileformat
data_onehot = data_io.reshape(len(data_io),41*4)
# data_onehot = data_onehot[0:201]

# Perform TNC encoding
tnc=read_nucleotide_sequences(data_path)
tnc = TNC(tnc)
tnc = np.array(tnc)
data_only4 = tnc[1:,2:]
# data_only4 = tnc[1:201,2:]
data_tnc = data_only4.astype(np.float)


# Define classifiers

xgb_cksnap = XGBClassifier()
xgb_onehot = XGBClassifier()
xgb_3mer = XGBClassifier()

lgbm_cksnap = lgb.LGBMClassifier()
lgbm_onehot = lgb.LGBMClassifier()
lgbm_3mer = lgb.LGBMClassifier()

svm_cksnap = SVC()
svm_onehot = SVC()
svm_3mer = SVC()

cat_cksnap = CatBoostClassifier()
cat_onehot = CatBoostClassifier()
cat_3mer = CatBoostClassifier()

rf_cksnap = RandomForestClassifier()
rf_onehot = RandomForestClassifier()
rf_3mer = RandomForestClassifier()


# xgb_cksnap = Booster()
# xgb_cksnap.load_model('/home/zeeshan/m5C_Mobeen/weights/danio_XGB_CKSNAP_1.h5')

# lgbm_cksnap = lgb.Booster(model_file='/home/zeeshan/m5C_Mobeen/weights/danio_LGBM_CKSNAP_1.h5')

# svm_cksnap = pickle.load(open('/home/zeeshan/m5C_Mobeen/weights/danio_SVM_CKSNAP_1.pickle', 'rb'))

# cat_cksnap = CatBoostClassifier()
# cat_cksnap.load_model('/home/zeeshan/m5C_Mobeen/weights/danio_CB_CKSNAP_1.h5')

# rf_cksnap = pickle.load(open('/home/zeeshan/m5C_Mobeen/weights/danio_RF_CKSNAP_1.pickle', 'rb'))
    
# # -----------
# xgb_onehot = Booster()
# xgb_onehot.load_model('/home/zeeshan/m5C_Mobeen/weights/danio_XGB_OH_1.h5')

# lgbm_onehot = lgb.Booster(model_file='/home/zeeshan/m5C_Mobeen/weights/danio_LGBM_OH_1.h5')

# svm_onehot = pickle.load(open('/home/zeeshan/m5C_Mobeen/weights/danio_SVM_OH_1.pickle', 'rb'))

# cat_onehot = CatBoostClassifier()
# cat_onehot.load_model('/home/zeeshan/m5C_Mobeen/weights/danio_CB_OH_1.h5')

# rf_onehot = pickle.load(open('/home/zeeshan/m5C_Mobeen/weights/danio_RF_OH_1.pickle', 'rb'))

# # -----------
# xgb_3mer = Booster()
# xgb_3mer.load_model('/home/zeeshan/m5C_Mobeen/weights/danio_XGB_3mer_1.h5')

# lgbm_3mer = lgb.Booster(model_file='/home/zeeshan/m5C_Mobeen/weights/danio_LGBM_3mer_1.h5')

# svm_3mer = pickle.load(open('/home/zeeshan/m5C_Mobeen/weights/danio_SVM_3mer_1.pickle', 'rb'))

# cat_3mer = CatBoostClassifier()
# cat_3mer.load_model('/home/zeeshan/m5C_Mobeen/weights/danio_CB_3mer_1.h5')

# rf_3mer = pickle.load(open('/home/zeeshan/m5C_Mobeen/weights/danio_RF_3mer_1.pickle', 'rb'))

# -----------
# Creating list of classifiers
classifiers_cksnap = [xgb_cksnap, lgbm_cksnap, svm_cksnap, cat_cksnap, rf_cksnap]
classifiers_onehot = [xgb_onehot, lgbm_onehot, svm_onehot, cat_onehot, rf_onehot]
classifiers_3mer = [xgb_3mer, lgbm_3mer, svm_3mer, cat_3mer, rf_3mer]

# Initialize lists to store evaluation metrics
test_accs = []
test_mccs = []
sens = []
spec = []
aucs = []
f1 = []
prec = []
rec = []
aucprc = []

all_test_preds = []
all_test_probs = []

length = len(data_onehot)

pos_lab = np.ones(int(length/2))
neg_lab = np.zeros(int(length/2))
labels = np.concatenate((pos_lab,neg_lab),axis=0)

c = list(zip(data_onehot, labels))
random.Random(100).shuffle(c)
data_io, labels_onehot = zip(*c)
data_onehot=np.asarray(data_io)
labels_onehot=np.asarray(labels_onehot)

c = list(zip(data_tnc, labels))
random.Random(100).shuffle(c)
data_io, labels_tnc = zip(*c)
data_tnc=np.asarray(data_io)
labels_tnc=np.asarray(labels_tnc)

c = list(zip(data_cksnap, labels))
random.Random(100).shuffle(c)
data_io, labels_cksnap = zip(*c)
data_cksnap=np.asarray(data_io)
labels_cksnap=np.asarray(labels_cksnap)




#%%


# Initialize lists to store predicted labels and probabilities for each classifier
all_test_preds = []  # Create an empty list to store predictions for each fold for all classifiers
all_test_probs = []  # Create an empty list to store probabilities for each fold for all classifiers
fold_test_accs = []
fold_test_mccs = []
fold_y_test = []
fold_y_test_preds = []

test_accs_xgb_onehot = []
test_accs_lgbm_onehot = []
test_accs_cat_onehot = []
test_accs_svm_onehot = []
test_accs_rf_onehot = []
test_accs_xgb_cksnap = []
test_accs_lgbm_cksnap = []
test_accs_cat_cksnap = []
test_accs_svm_cksnap = []
test_accs_rf_cksnap = []
test_accs_xgb_3mer = []
test_accs_lgbm_3mer = []
test_accs_cat_3mer = []
test_accs_svm_3mer = []
test_accs_rf_3mer = []
mean_acc_xgb = []
mean_acc_lgbm = []
mean_acc_cat = []
mean_acc_svm = []
mean_acc_rf = []

all_y_tests = []
ypred_onehots_xgb = []
yprobs_onehots_xgb = []
ypred_onehots_lgbm = []
yprobs_onehots_lgbm = []
ypred_onehots_cat = []
yprobs_onehots_cat = []
ypred_onehots_svm = []
yprobs_onehots_svm = []
ypred_onehots_rf = []
yprobs_onehots_rf = []

ypred_cksnaps_xgb = []
yprobs_cksnaps_xgb = []
ypred_cksnaps_lgbm = []
yprobs_cksnaps_lgbm = []
ypred_cksnaps_cat = []
yprobs_cksnaps_cat = []
ypred_cksnaps_svm = []
yprobs_cksnaps_svm = []
ypred_cksnaps_rf = []
yprobs_cksnaps_rf = []

ypred_3mers_xgb = []
yprobs_3mers_xgb = []
ypred_3mers_lgbm = []
yprobs_3mers_lgbm = []
ypred_3mers_cat = []
yprobs_3mers_cat = []
ypred_3mers_svm = []
yprobs_3mers_svm = []
ypred_3mers_rf = []
yprobs_3mers_rf = []                
                
# folds = 5
# kf = KFold(n_splits=folds, shuffle=True, random_state=4)

# Initialize lists to store results for all folds
all_y_test = []
all_test_probs = []
y_onehots = []


for classifier in classifiers_onehot:
        if classifier.__class__.__name__ == 'CatBoostClassifier':
            # X_train_catboost = pd.DataFrame(data_onehot)
            X_test_catboost = pd.DataFrame(data_onehot)
            classifier.load_model('/home/zeeshan/m5C_Mobeen/weights/danio_CB_OH_1.h5')
            # classifier.fit(X_train_catboost, y_train)
            y_pred = classifier.predict(X_test_catboost)
            y_prob = classifier.predict_proba(X_test_catboost)[:, 1]
            acc_cat_onehot = metrics.accuracy_score(labels_onehot, y_pred)
            print('CAT Accuracy OH: ', acc_cat_onehot)
            test_accs_cat_onehot.append(acc_cat_onehot) 
            ypred_onehots_cat.append(y_pred) 
            
        elif classifier.__class__.__name__ == 'XGBClassifier':
            classifier.load_model('/home/zeeshan/m5C_Mobeen/weights/danio_XGB_OH_1.h5')
            y_pred = classifier.predict(data_onehot)
            y_prob = classifier.predict_proba(data_onehot)[:, 1]
            acc_xgb_onehot = metrics.accuracy_score(labels_onehot, y_pred)
            print('XGB Accuracy OH: ', acc_xgb_onehot)
            test_accs_xgb_onehot.append(acc_xgb_onehot) # for mean acc
            ypred_onehots_xgb.append(y_pred) # all 5 fold preds
            yprobs_onehots_xgb.append(y_prob)
            # all_y_tests.append(y_test) # combining all y_tests for ensemble; no need to do for all classifiers
            # classifier.save_model('/home/zeeshan/m5C_Mobeen/weights/danio_XGB_OH_'+str(i+1)+'.h5')
  
        elif classifier.__class__.__name__ == 'LGBMClassifier':
            # lgbm_cksnap = lgb.Booster(model_file='/home/zeeshan/m5C_Mobeen/weights/danio_LGBM_CKSNAP_1.h5')
            classifier = lgb.Booster(model_file='/home/zeeshan/m5C_Mobeen/weights/danio_LGBM_OH_1.h5')
            classifier.booster_ = classifier
            y_pred = classifier.predict(data_onehot)
            # y_prob = classifier.predict_proba(data_onehot)[:, 1]
            threshold = 0.5  # Adjust this threshold based on your problem
            y_pred = (y_prob > threshold).astype(int)
            acc_lgbm_onehot = metrics.accuracy_score(labels_onehot, y_pred)
            print('LGBM Accuracy OH: ', acc_lgbm_onehot)
            test_accs_lgbm_onehot.append(acc_lgbm_onehot)
            ypred_onehots_lgbm.append(y_pred)
            yprobs_onehots_lgbm.append(y_prob)
            # classifier.booster_.save_model('/home/zeeshan/m5C_Mobeen/weights/danio_LGBM_OH_'+str(i+1)+'.h5')
            
        elif classifier.__class__.__name__ == 'SVC':
            svc_model = '/home/zeeshan/m5C_Mobeen/weights/danio_SVM_OH_1.pickle'
            with open(svc_model, 'rb') as model_file:
                classifier = pickle.load(model_file)
                       
            y_pred = classifier.predict(data_onehot)
            # y_prob = classifier.predict_proba(X_test)[:, 1]
            acc_svm_onehot = metrics.accuracy_score(labels_onehot, y_pred)
            print('SVM Accuracy OH: ', acc_svm_onehot)
            test_accs_svm_onehot.append(acc_svm_onehot)
            ypred_onehots_svm.append(y_pred)
            

        elif classifier.__class__.__name__ == 'RandomForestClassifier':
            rf_model = '/home/zeeshan/m5C_Mobeen/weights/danio_RF_OH_1.pickle'
            with open(svc_model, 'rb') as model_file:
                classifier = pickle.load(model_file)
                
            y_pred = classifier.predict(data_onehot)
            # y_prob = classifier.predict_proba(X_test)[:, 1]
            acc_rf_onehot = metrics.accuracy_score(labels_onehot, y_pred)
            print('RF Accuracy OH: ', acc_rf_onehot)
            test_accs_rf_onehot.append(acc_rf_onehot)
            ypred_onehots_rf.append(y_pred)
            # yprobs_onehots_rf.append(y_prob)
    
    
    
    
for classifier in classifiers_cksnap:
        if classifier.__class__.__name__ == 'CatBoostClassifier':
            # X_train_catboost = pd.DataFrame(data_onehot)
            X_test_catboost = pd.DataFrame(data_cksnap)
            classifier.load_model('/home/zeeshan/m5C_Mobeen/weights/danio_CB_CKSNAP_1.h5')
            # classifier.fit(X_train_catboost, y_train)
            y_pred = classifier.predict(X_test_catboost)
            y_prob = classifier.predict_proba(X_test_catboost)[:, 1]
            acc_cat_cksnap = metrics.accuracy_score(labels_cksnap, y_pred)
            print('CAT Accuracy CKSNAP: ', acc_cat_cksnap)
            test_accs_cat_onehot.append(acc_cat_cksnap) 
            ypred_cksnaps_cat.append(y_pred) 
            
        elif classifier.__class__.__name__ == 'XGBClassifier':
            classifier.load_model('/home/zeeshan/m5C_Mobeen/weights/danio_XGB_CKSNAP_1.h5')
            y_pred = classifier.predict(data_cksnap)
            y_prob = classifier.predict_proba(data_cksnap)[:, 1]
            acc_xgb_cksnap = metrics.accuracy_score(labels_cksnap, y_pred)
            print('XGB Accuracy CKSNAP: ', acc_xgb_cksnap)
            test_accs_xgb_onehot.append(acc_xgb_cksnap) # for mean acc
            ypred_cksnaps_xgb.append(y_pred) # all 5 fold preds
            yprobs_cksnaps_xgb.append(y_prob)
            # all_y_tests.append(y_test) # combining all y_tests for ensemble; no need to do for all classifiers
            # classifier.save_model('/home/zeeshan/m5C_Mobeen/weights/danio_XGB_OH_'+str(i+1)+'.h5')
  
        elif classifier.__class__.__name__ == 'LGBMClassifier':
            # lgbm_cksnap = lgb.Booster(model_file='/home/zeeshan/m5C_Mobeen/weights/danio_LGBM_CKSNAP_1.h5')
            classifier = lgb.Booster(model_file='/home/zeeshan/m5C_Mobeen/weights/danio_LGBM_CKSNAP_1.h5')
            classifier.booster_ = classifier
            y_pred = classifier.predict(data_cksnap)
            # y_prob = classifier.predict_proba(data_onehot)[:, 1]
            threshold = 0.5  # Adjust this threshold based on your problem
            y_pred = (y_prob > threshold).astype(int)
            acc_lgbm_cksnap = metrics.accuracy_score(labels_cksnap, y_pred)
            print('LGBM Accuracy CKSNAP: ', acc_lgbm_cksnap)
            test_accs_lgbm_onehot.append(acc_lgbm_cksnap)
            ypred_cksnaps_lgbm.append(y_pred)
            yprobs_cksnaps_lgbm.append(y_prob)
            # classifier.booster_.save_model('/home/zeeshan/m5C_Mobeen/weights/danio_LGBM_OH_'+str(i+1)+'.h5')
            
        elif classifier.__class__.__name__ == 'SVC':
            svc_model = '/home/zeeshan/m5C_Mobeen/weights/danio_SVM_CKSNAP_1.pickle'
            with open(svc_model, 'rb') as model_file:
                classifier = pickle.load(model_file)
                       
            y_pred = classifier.predict(data_cksnap)
            # y_prob = classifier.predict_proba(X_test)[:, 1]
            acc_svm_cksnap = metrics.accuracy_score(labels_cksnap, y_pred)
            print('SVM Accuracy CKSNAP: ', acc_svm_cksnap)
            test_accs_svm_onehot.append(acc_svm_cksnap)
            ypred_cksnaps_svm.append(y_pred)
            

        elif classifier.__class__.__name__ == 'RandomForestClassifier':
            rf_model = '/home/zeeshan/m5C_Mobeen/weights/danio_RF_CKSNAP_1.pickle'
            with open(svc_model, 'rb') as model_file:
                classifier = pickle.load(model_file)
                
            y_pred = classifier.predict(data_cksnap)
            # y_prob = classifier.predict_proba(X_test)[:, 1]
            acc_rf_cksnap = metrics.accuracy_score(labels_cksnap, y_pred)
            print('RF Accuracy CKSNAP: ', acc_rf_cksnap)
            test_accs_rf_onehot.append(acc_rf_cksnap)
            ypred_cksnaps_rf.append(y_pred)
            # yprobs_onehots_rf.append(y_prob)
            
            
            
for classifier in classifiers_3mer:
        if classifier.__class__.__name__ == 'CatBoostClassifier':
            # X_train_catboost = pd.DataFrame(data_onehot)
            X_test_catboost = pd.DataFrame(data_tnc)
            classifier.load_model('/home/zeeshan/m5C_Mobeen/weights/danio_CB_3mer_1.h5')
            # classifier.fit(X_train_catboost, y_train)
            y_pred = classifier.predict(X_test_catboost)
            y_prob = classifier.predict_proba(X_test_catboost)[:, 1]
            acc_cat_3mer = metrics.accuracy_score(labels_tnc, y_pred)
            print('CAT Accuracy 3mer: ', acc_cat_3mer)
            test_accs_cat_onehot.append(acc_cat_3mer) 
            ypred_3mers_cat.append(y_pred) 
            
        elif classifier.__class__.__name__ == 'XGBClassifier':
            classifier.load_model('/home/zeeshan/m5C_Mobeen/weights/danio_XGB_3mer_1.h5')
            y_pred = classifier.predict(data_tnc)
            y_prob = classifier.predict_proba(data_tnc)[:, 1]
            acc_xgb_3mer = metrics.accuracy_score(labels_tnc, y_pred)
            print('XGB Accuracy 3mer: ', acc_xgb_3mer)
            test_accs_xgb_onehot.append(acc_xgb_3mer) # for mean acc
            ypred_3mers_xgb.append(y_pred) # all 5 fold preds
            yprobs_3mers_xgb.append(y_prob)
            # all_y_tests.append(y_test) # combining all y_tests for ensemble; no need to do for all classifiers
            # classifier.save_model('/home/zeeshan/m5C_Mobeen/weights/danio_XGB_OH_'+str(i+1)+'.h5')
  
        elif classifier.__class__.__name__ == 'LGBMClassifier':
            # lgbm_cksnap = lgb.Booster(model_file='/home/zeeshan/m5C_Mobeen/weights/danio_LGBM_CKSNAP_1.h5')
            classifier = lgb.Booster(model_file='/home/zeeshan/m5C_Mobeen/weights/danio_LGBM_3mer_1.h5')
            classifier.booster_ = classifier
            y_pred = classifier.predict(data_tnc)
            # y_prob = classifier.predict_proba(data_onehot)[:, 1]
            threshold = 0.5  # Adjust this threshold based on your problem
            y_pred = (y_prob > threshold).astype(int)
            acc_lgbm_3mer = metrics.accuracy_score(labels_tnc, y_pred)
            print('LGBM Accuracy 3mer: ', acc_lgbm_3mer)
            test_accs_lgbm_onehot.append(acc_lgbm_3mer)
            ypred_3mers_lgbm.append(y_pred)
            yprobs_3mers_lgbm.append(y_prob)
            # classifier.booster_.save_model('/home/zeeshan/m5C_Mobeen/weights/danio_LGBM_OH_'+str(i+1)+'.h5')
            
        elif classifier.__class__.__name__ == 'SVC':
            svc_model = '/home/zeeshan/m5C_Mobeen/weights/danio_SVM_3mer_1.pickle'
            with open(svc_model, 'rb') as model_file:
                classifier = pickle.load(model_file)
                       
            y_pred = classifier.predict(data_tnc)
            # y_prob = classifier.predict_proba(X_test)[:, 1]
            acc_svm_3mer = metrics.accuracy_score(labels_tnc, y_pred)
            print('SVM Accuracy 3mer: ', acc_svm_3mer)
            test_accs_svm_onehot.append(acc_svm_3mer)
            ypred_3mers_svm.append(y_pred)
            

        elif classifier.__class__.__name__ == 'RandomForestClassifier':
            rf_model = '/home/zeeshan/m5C_Mobeen/weights/danio_RF_3mer_1.pickle'
            with open(svc_model, 'rb') as model_file:
                classifier = pickle.load(model_file)
                
            y_pred = classifier.predict(data_tnc)
            # y_prob = classifier.predict_proba(X_test)[:, 1]
            acc_rf_3mer = metrics.accuracy_score(labels_tnc, y_pred)
            print('RF Accuracy 3mer: ', acc_rf_3mer)
            test_accs_rf_onehot.append(acc_rf_3mer)
            ypred_3mers_rf.append(y_pred)
            # yprobs_onehots_rf.append(y_prob)
#%%            
              
                
mean_acc_xgb_onehot = mean(test_accs_xgb_onehot)
mean_acc_lgbm_onehot = mean(test_accs_lgbm_onehot)
mean_acc_cat_onehot = mean(test_accs_cat_onehot)
mean_acc_svm_onehot = mean(test_accs_svm_onehot)
mean_acc_rf_onehot = mean(test_accs_rf_onehot)

mean_acc_xgb_cksnap = mean(test_accs_xgb_cksnap)
mean_acc_lgbm_cksnap = mean(test_accs_lgbm_cksnap)
mean_acc_cat_cksnap = mean(test_accs_cat_cksnap)
mean_acc_svm_cksnap = mean(test_accs_svm_cksnap)
mean_acc_rf_cksnap = mean(test_accs_rf_cksnap)

mean_acc_xgb_3mer = mean(test_accs_xgb_3mer)
mean_acc_lgbm_3mer = mean(test_accs_lgbm_3mer)
mean_acc_cat_3mer = mean(test_accs_cat_3mer)
mean_acc_svm_3mer = mean(test_accs_svm_3mer)
mean_acc_rf_3mer = mean(test_accs_rf_3mer)


### Combining all 5 folds predictions to apply ensemble
y_test_preds_xgb_onehot = np.concatenate(ypred_onehots_xgb, axis=0)
y_test_preds_lgbm_onehot = np.concatenate(ypred_onehots_lgbm, axis=0)
y_test_preds_cat_onehot = np.concatenate(ypred_onehots_cat, axis=0)
y_test_preds_svm_onehot = np.concatenate(ypred_onehots_svm, axis=0)
y_test_preds_rf_onehot = np.concatenate(ypred_onehots_rf, axis=0)

y_test_preds_xgb_cksnap = np.concatenate(ypred_cksnaps_xgb, axis=0)
y_test_preds_lgbm_cksnap = np.concatenate(ypred_cksnaps_lgbm, axis=0)
y_test_preds_cat_cksnap = np.concatenate(ypred_cksnaps_cat, axis=0)
y_test_preds_svm_cksnap = np.concatenate(ypred_cksnaps_svm, axis=0)
y_test_preds_rf_cksnap = np.concatenate(ypred_cksnaps_rf, axis=0)

y_test_preds_xgb_3mer = np.concatenate(ypred_3mers_xgb, axis=0)
y_test_preds_lgbm_3mer = np.concatenate(ypred_3mers_lgbm, axis=0)
y_test_preds_cat_3mer = np.concatenate(ypred_3mers_cat, axis=0)
y_test_preds_svm_3mer = np.concatenate(ypred_3mers_svm, axis=0)
y_test_preds_rf_3mer = np.concatenate(ypred_3mers_rf, axis=0)

# ### Combining all 5 folds probs to apply ensemble ### -------- Just to check using probs
# y_test_probs_xgb_onehot = np.concatenate(yprobs_onehots_xgb, axis=0)
# y_test_probs_lgbm_onehot = np.concatenate(yprobs_onehots_lgbm, axis=0)
# y_test_probs_cat_onehot = np.concatenate(yprobs_onehots_cat, axis=0)
# # y_test_probs_svm_onehot = np.concatenate(yprobs_onehots_svm, axis=0)
# y_test_probs_rf_onehot = np.concatenate(yprobs_onehots_rf, axis=0)

# y_test_probs_xgb_cksnap = np.concatenate(yprobs_cksnaps_xgb, axis=0)
# y_test_probs_lgbm_cksnap = np.concatenate(yprobs_cksnaps_lgbm, axis=0)
# y_test_probs_cat_cksnap = np.concatenate(yprobs_cksnaps_cat, axis=0)
# # y_test_probs_svm_cksnap = np.concatenate(yprobs_cksnaps_svm, axis=0)
# y_test_probs_rf_cksnap = np.concatenate(yprobs_cksnaps_rf, axis=0)

# y_test_probs_xgb_3mer = np.concatenate(yprobs_3mers_xgb, axis=0)
# y_test_probs_lgbm_3mer = np.concatenate(yprobs_3mers_lgbm, axis=0)
# y_test_probs_cat_3mer = np.concatenate(yprobs_3mers_cat, axis=0)
# # y_test_probs_svm_3mer = np.concatenate(yprobs_3mers_svm, axis=0)
# y_test_probs_rf_3mer = np.concatenate(yprobs_3mers_rf, axis=0)


# all_y_testss = np.concatenate(all_y_tests, axis=0)
all_y_preds = np.column_stack((y_test_preds_xgb_onehot, y_test_preds_lgbm_onehot, y_test_preds_cat_onehot, y_test_preds_svm_onehot, y_test_preds_rf_onehot,
                               y_test_preds_xgb_cksnap, y_test_preds_lgbm_cksnap, y_test_preds_cat_cksnap, y_test_preds_svm_cksnap, y_test_preds_rf_cksnap,
                               y_test_preds_xgb_3mer, y_test_preds_lgbm_3mer, y_test_preds_cat_3mer, y_test_preds_svm_3mer, y_test_preds_rf_3mer))

# all_y_probs = np.column_stack((y_test_probs_xgb_onehot, y_test_probs_lgbm_onehot, y_test_probs_cat_onehot, y_test_probs_rf_onehot,
#                                y_test_probs_xgb_cksnap, y_test_probs_lgbm_cksnap, y_test_probs_cat_cksnap, y_test_probs_rf_cksnap,
#                                y_test_probs_xgb_3mer, y_test_probs_lgbm_3mer, y_test_probs_cat_3mer, y_test_probs_rf_3mer))


#%%

from sklearn.linear_model import LogisticRegression
meta_classifier = LogisticRegression(random_state=0)

meta_classifier = pickle.load(open('/home/zeeshan/m5C_Mobeen/weights/danio_META_2.pickle', 'rb'))
y_pred_ens = meta_classifier.predict(all_y_preds)
acc_ens = metrics.accuracy_score(y_pred_ens, labels_onehot)
print('Ensemble Acc: ', acc_ens)


mcc_ens = metrics.matthews_corrcoef(y_pred_ens, labels_onehot)
print('Ensemble Mcc: ', mcc_ens)


confusion = confusion_matrix(y_pred_ens, labels_onehot)
TN, FP, FN, TP = confusion.ravel()
sensitivity = TP / float(TP + FN)
print('Ensemble Sen: ', sensitivity)

specificity = TN / float(TN + FP)
print('Ensemble Spe: ', specificity)

F1Score = (2 * TP) / float(2 * TP + FP + FN)
print('Ensemble F1: ', F1Score)

precision = TP / float(TP + FP)
print('Ensemble Pre: ', precision)

recall = TP / float(TP + FN)
print('Ensemble Recall: ', recall)

#AUC
y_pred_prob = meta_classifier.predict_proba(all_y_preds) #----
y_probs = y_pred_prob[:,1]
ROCArea = roc_auc_score(labels_onehot, y_probs)
print('Ensemble RoC: ', ROCArea)
    
    


   










