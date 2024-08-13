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

# Read and preprocess the data
data_path = '/home/zeeshan/m5C_Mobeen/datasets/train_danio.fasta'

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
xgb_cksnap = XGBClassifier(learning_rate=0.017965145536083, max_depth=11, 
                min_child_weight=1, gamma=0.17426232112256684, 
                colsample_bytree=0.1916286359142644, n_estimators=625, seed=14)

lgbm_cksnap = lgb.LGBMClassifier(learning_rate=0.011038126658226765, max_depth=12,
                min_child_samples=5, min_child_weight=0.4318851550452938, min_split_gain=0.025127530387291663,
                n_estimators=471, num_leaves=31) 

svm_cksnap = SVC(C=2, kernel='rbf', gamma='scale')

cat_cksnap = CatBoostClassifier(learning_rate=0.12441112053758573, max_depth=6,iterations=100,
                                  boosting_type='Plain')

rf_cksnap = RandomForestClassifier(max_depth=80, bootstrap=False, max_features = 'sqrt',
                                    n_estimators=400, min_samples_split=3,random_state=0)
    
# -----------
xgb_onehot = XGBClassifier(learning_rate=0.031056714993020405, max_depth=16, 
                min_child_weight=15, gamma=0.6839886016733135, 
                colsample_bytree=0.11295327041737442, n_estimators=1140, seed=14)

lgbm_onehot = lgb.LGBMClassifier(learning_rate=0.018360057115182342, max_depth=12,
                min_child_samples=13, min_child_weight=0.17787985358276698, min_split_gain=0.028999977187230613,
                n_estimators=482, num_leaves=24) 

svm_onehot = SVC(C=1, kernel='rbf', gamma='scale')

cat_onehot = CatBoostClassifier(learning_rate=0.25659939142290733, max_depth=4,iterations=90,
                                  boosting_type='Plain')

rf_onehot = RandomForestClassifier(max_depth=30, bootstrap=False, max_features = 'log2',
                                    n_estimators=600, min_samples_split=6,random_state=0)

# -----------
xgb_3mer = XGBClassifier(learning_rate=0.010474246167072487, max_depth=14, 
                min_child_weight=1, gamma=0.6516423460350472, 
                colsample_bytree=0.2464115690220173, n_estimators=833, seed=14)

lgbm_3mer = lgb.LGBMClassifier(learning_rate=0.0753730090287657, max_depth=13,
                min_child_samples=11, min_child_weight=0.48660326897207984, min_split_gain=0.009254681493502266,
                n_estimators=357, num_leaves=49) 

svm_3mer = SVC(C=1, kernel='rbf', gamma='scale')

cat_3mer = CatBoostClassifier(learning_rate=0.22322424314778602, max_depth=5,iterations=80,
                                  boosting_type='Plain')

rf_3mer = RandomForestClassifier(max_depth=30, bootstrap=False, max_features = 'log2',
                                    n_estimators=1400, min_samples_split=2,random_state=0)

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
                
folds = 5
kf = KFold(n_splits=folds, shuffle=True, random_state=4)

# Initialize lists to store results for all folds
all_y_test = []
all_test_probs = []
y_onehots = []


for i, (train_index, test_index) in enumerate(kf.split(data_onehot, labels_onehot)):
    X_train_onehott, X_test_onehot = data_onehot[train_index], data_onehot[test_index]
    X_train_tncc, X_test_tnc = data_tnc[train_index], data_tnc[test_index]
    X_train_cksnapp, X_test_cksnap = data_cksnap[train_index], data_cksnap[test_index]
    
    y_train_onehott, y_test_onehot = labels_onehot[train_index], labels_onehot[test_index]
    y_train_tncc, y_test_tnc = labels_tnc[train_index], labels_tnc[test_index]
    y_train_cksnapp, y_test_cksnap = labels_cksnap[train_index], labels_cksnap[test_index]
    
    
    X_train_onehot, X_validation_onehot, y_train_onehot, y_validation_onehot = train_test_split(X_train_onehott, y_train_onehott, test_size=0.01, random_state=92, shuffle=True)
    X_train_tnc, X_validation_tnc, y_train_tnc, y_validation_tnc = train_test_split(X_train_tncc, y_train_tncc, test_size=0.01, random_state=92, shuffle=True)
    X_train_cksnap, X_validation_cksnap, y_train_cksnap, y_validation_cksnap = train_test_split(X_train_cksnapp, y_train_cksnapp, test_size=0.01, random_state=92, shuffle=True)
        

    # Initialize lists to store predicted labels and probabilities for each classifier
    y_test_preds_list = []
    y_test_probs_list = []

    for classifier in classifiers_onehot:
        for X_train, X_test, y_train, y_test in [(X_train_onehot, X_test_onehot, y_train_onehot, y_test_onehot)]:
            # print(X_train.shape, y_train.shape)
            print('Individual Classifiers fold: ', i)
            if classifier.__class__.__name__ == 'CatBoostClassifier':
                X_train_catboost = pd.DataFrame(X_train)
                X_test_catboost = pd.DataFrame(X_test)
                classifier.fit(X_train_catboost, y_train)
                y_pred = classifier.predict(X_test_catboost)
                y_prob = classifier.predict_proba(X_test_catboost)[:, 1]
                acc_cat_onehot = metrics.accuracy_score(y_test, y_pred)
                print('CAT Accuracy OH: ', acc_cat_onehot)
                test_accs_cat_onehot.append(acc_cat_onehot) 
                ypred_onehots_cat.append(y_pred) 
                yprobs_onehots_cat.append(y_prob)
                # classifier.save_model('/home/zeeshan/m5C_Mobeen/weights/danio_CB_OH_'+str(i+1)+'.h5')
                
            elif classifier.__class__.__name__ == 'XGBClassifier':
                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test)
                y_prob = classifier.predict_proba(X_test)[:, 1]
                acc_xgb_onehot = metrics.accuracy_score(y_test, y_pred)
                print('XGB Accuracy OH: ', acc_xgb_onehot)
                test_accs_xgb_onehot.append(acc_xgb_onehot) # for mean acc
                ypred_onehots_xgb.append(y_pred) # all 5 fold preds
                yprobs_onehots_xgb.append(y_prob)
                all_y_tests.append(y_test) # combining all y_tests for ensemble; no need to do for all classifiers
                # classifier.save_model('/home/zeeshan/m5C_Mobeen/weights/danio_XGB_OH_'+str(i+1)+'.h5')
  
            elif classifier.__class__.__name__ == 'LGBMClassifier':
                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test)
                y_prob = classifier.predict_proba(X_test)[:, 1]
                acc_lgbm_onehot = metrics.accuracy_score(y_test, y_pred)
                print('LGBM Accuracy OH: ', acc_lgbm_onehot)
                test_accs_lgbm_onehot.append(acc_lgbm_onehot)
                ypred_onehots_lgbm.append(y_pred)
                yprobs_onehots_lgbm.append(y_prob)
                # classifier.booster_.save_model('/home/zeeshan/m5C_Mobeen/weights/danio_LGBM_OH_'+str(i+1)+'.h5')
                
            elif classifier.__class__.__name__ == 'SVC':
                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test)
                # y_prob = classifier.predict_proba(X_test)[:, 1]
                acc_svm_onehot = metrics.accuracy_score(y_test, y_pred)
                print('SVM Accuracy OH: ', acc_svm_onehot)
                test_accs_svm_onehot.append(acc_svm_onehot)
                ypred_onehots_svm.append(y_pred)
                # yprobs_onehots_svm.append(y_prob)
                # svc_model = '/home/zeeshan/m5C_Mobeen/weights/danio_SVM_OH_'+str(i+1)+'.pickle'
                # with open(svc_model, 'wb') as model_file:
                #     pickle.dump(classifier, model_file)
    
            elif classifier.__class__.__name__ == 'RandomForestClassifier':
                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test)
                y_prob = classifier.predict_proba(X_test)[:, 1]
                acc_rf_onehot = metrics.accuracy_score(y_test, y_pred)
                print('RF Accuracy OH: ', acc_rf_onehot)
                test_accs_rf_onehot.append(acc_rf_onehot)
                ypred_onehots_rf.append(y_pred)
                yprobs_onehots_rf.append(y_prob)
                # classifier.save('/home/zeeshan/m5C_Mobeen/weights/human_RF_OH_'+str(i+1)+'.h5')
                # rf_model = '/home/zeeshan/m5C_Mobeen/weights/danio_RF_OH_'+str(i+1)+'.pickle'
                # with open(rf_model, 'wb') as model_file:
                #     pickle.dump(classifier, model_file)
                
        
    for classifier in classifiers_cksnap:
        for X_train, X_test, y_train, y_test in [(X_train_cksnap, X_test_cksnap, y_train_cksnap, y_test_cksnap)]:   
        
            # print(X_train.shape, y_train.shape)
            # print('fold', i)
            if classifier.__class__.__name__ == 'CatBoostClassifier':
                X_train_catboost = pd.DataFrame(X_train)
                X_test_catboost = pd.DataFrame(X_test)
                classifier.fit(X_train_catboost, y_train)
                y_pred = classifier.predict(X_test_catboost)
                y_prob = classifier.predict_proba(X_test_catboost)[:, 1]
                acc_cat_cksnap = metrics.accuracy_score(y_test, y_pred)
                print('CAT Accuracy CKSNAP: ', acc_cat_cksnap)
                test_accs_cat_cksnap.append(acc_cat_cksnap) 
                ypred_cksnaps_cat.append(y_pred) 
                yprobs_cksnaps_cat.append(y_prob)
                # classifier.save_model('/home/zeeshan/m5C_Mobeen/weights/danio_CB_CKSNAP_'+str(i+1)+'.h5')
                
            elif classifier.__class__.__name__ == 'XGBClassifier':
                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test)
                y_prob = classifier.predict_proba(X_test)[:, 1]
                acc_xgb_cksnap = accuracy_score(y_pred, y_test)
                print('XGB Accuracy CKSNAP: ', acc_xgb_cksnap)
                test_accs_xgb_cksnap.append(acc_xgb_cksnap)
                ypred_cksnaps_xgb.append(y_pred)
                yprobs_cksnaps_xgb.append(y_prob)
                # classifier.save_model('/home/zeeshan/m5C_Mobeen/weights/danio_XGB_CKSNAP_'+str(i+1)+'.h5')
            
            elif classifier.__class__.__name__ == 'LGBMClassifier':
                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test)
                y_prob = classifier.predict_proba(X_test)[:, 1]
                acc_lgbm_cksnap = accuracy_score(y_pred, y_test)
                print('LGBM Accuracy CKSNAP: ', acc_lgbm_cksnap)
                test_accs_lgbm_cksnap.append(acc_lgbm_cksnap)
                ypred_cksnaps_lgbm.append(y_pred)
                yprobs_cksnaps_lgbm.append(y_prob)
                # classifier.booster_.save_model('/home/zeeshan/m5C_Mobeen/weights/danio_LGBM_CKSNAP_'+str(i+1)+'.h5')
                
            elif classifier.__class__.__name__ == 'SVC':
                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test)
                # y_prob = classifier.predict_proba(X_test)[:, 1]
                acc_svm_cksnap = metrics.accuracy_score(y_test, y_pred)
                print('SVM Accuracy CKSNAP: ', acc_svm_cksnap)
                test_accs_svm_cksnap.append(acc_svm_cksnap)
                ypred_cksnaps_svm.append(y_pred)
                # yprobs_onehots_svm.append(y_prob)
                # svc_model = '/home/zeeshan/m5C_Mobeen/weights/danio_SVM_CKSNAP_'+str(i+1)+'.pickle'
                # with open(svc_model, 'wb') as model_file:
                #     pickle.dump(classifier, model_file)
                
            elif classifier.__class__.__name__ == 'RandomForestClassifier':
                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test)
                y_prob = classifier.predict_proba(X_test)[:, 1]
                acc_rf_cksnap = metrics.accuracy_score(y_test, y_pred)
                print('RF Accuracy CKSNAP: ', acc_rf_cksnap)
                test_accs_rf_cksnap.append(acc_rf_cksnap)
                ypred_cksnaps_rf.append(y_pred)
                yprobs_cksnaps_rf.append(y_prob)
                # rf_model = '/home/zeeshan/m5C_Mobeen/weights/danio_RF_CKSNAP_'+str(i+1)+'.pickle'
                # with open(rf_model, 'wb') as model_file:
                #     pickle.dump(classifier, model_file)
                
    for classifier in classifiers_3mer:
        for X_train, X_test, y_train, y_test in [(X_train_tnc, X_test_tnc, y_train_tnc, y_test_tnc)]:   
        
            # print(X_train.shape, y_train.shape)
            # print('fold', i)
            if classifier.__class__.__name__ == 'CatBoostClassifier':
                X_train_catboost = pd.DataFrame(X_train)
                X_test_catboost = pd.DataFrame(X_test)
                classifier.fit(X_train_catboost, y_train)
                y_pred = classifier.predict(X_test_catboost)
                y_prob = classifier.predict_proba(X_test_catboost)[:, 1]
                acc_cat_3mer = metrics.accuracy_score(y_test, y_pred)
                print('CAT Accuracy 3mer: ', acc_cat_3mer)
                test_accs_cat_3mer.append(acc_cat_3mer) 
                ypred_3mers_cat.append(y_pred) 
                yprobs_3mers_cat.append(y_prob)
                # classifier.save_model('/home/zeeshan/m5C_Mobeen/weights/danio_CB_3mer_'+str(i+1)+'.h5')
                
            elif classifier.__class__.__name__ == 'XGBClassifier':
                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test)
                y_prob = classifier.predict_proba(X_test)[:, 1]
                acc_xgb_3mer = accuracy_score(y_pred, y_test)
                print('XGB Accuracy 3mer: ', acc_xgb_3mer)
                test_accs_xgb_3mer.append(acc_xgb_3mer)
                ypred_3mers_xgb.append(y_pred)
                yprobs_3mers_xgb.append(y_prob)
                # classifier.save_model('/home/zeeshan/m5C_Mobeen/weights/danio_XGB_3mer_'+str(i+1)+'.h5')
            
            elif classifier.__class__.__name__ == 'LGBMClassifier':
                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test)
                y_prob = classifier.predict_proba(X_test)[:, 1]
                acc_lgbm_3mer = accuracy_score(y_pred, y_test)
                print('LGBM Accuracy 3mer: ', acc_lgbm_3mer)
                test_accs_lgbm_3mer.append(acc_lgbm_3mer)
                ypred_3mers_lgbm.append(y_pred)
                yprobs_3mers_lgbm.append(y_prob)
                # classifier.booster_.save_model('/home/zeeshan/m5C_Mobeen/weights/danio_LGBM_3mer_'+str(i+1)+'.h5')
                
            elif classifier.__class__.__name__ == 'SVC':
                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test)
                # y_prob = classifier.predict_proba(X_test)[:, 1]
                acc_svm_3mer = metrics.accuracy_score(y_test, y_pred)
                print('SVM Accuracy 3mer: ', acc_svm_3mer)
                test_accs_svm_3mer.append(acc_svm_3mer)
                ypred_3mers_svm.append(y_pred)
                # yprobs_onehots_svm.append(y_prob)
                # svc_model = '/home/zeeshan/m5C_Mobeen/weights/danio_SVM_3mer_'+str(i+1)+'.pickle'
                # with open(svc_model, 'wb') as model_file:
                #     pickle.dump(classifier, model_file)
                
            elif classifier.__class__.__name__ == 'RandomForestClassifier':
                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test)
                y_prob = classifier.predict_proba(X_test)[:, 1]
                acc_rf_3mer = metrics.accuracy_score(y_test, y_pred)
                print('RF Accuracy 3mer: ', acc_rf_3mer)
                test_accs_rf_3mer.append(acc_rf_3mer)
                ypred_3mers_rf.append(y_pred)
                yprobs_3mers_rf.append(y_prob)
                # rf_model = '/home/zeeshan/m5C_Mobeen/weights/danio_RF_3mer_'+str(i+1)+'.pickle'
                # with open(rf_model, 'wb') as model_file:
                #     pickle.dump(classifier, model_file)

                
                
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

### Combining all 5 folds probs to apply ensemble ### -------- Just to check using probs
y_test_probs_xgb_onehot = np.concatenate(yprobs_onehots_xgb, axis=0)
y_test_probs_lgbm_onehot = np.concatenate(yprobs_onehots_lgbm, axis=0)
y_test_probs_cat_onehot = np.concatenate(yprobs_onehots_cat, axis=0)
# y_test_probs_svm_onehot = np.concatenate(yprobs_onehots_svm, axis=0)
y_test_probs_rf_onehot = np.concatenate(yprobs_onehots_rf, axis=0)

y_test_probs_xgb_cksnap = np.concatenate(yprobs_cksnaps_xgb, axis=0)
y_test_probs_lgbm_cksnap = np.concatenate(yprobs_cksnaps_lgbm, axis=0)
y_test_probs_cat_cksnap = np.concatenate(yprobs_cksnaps_cat, axis=0)
# y_test_probs_svm_cksnap = np.concatenate(yprobs_cksnaps_svm, axis=0)
y_test_probs_rf_cksnap = np.concatenate(yprobs_cksnaps_rf, axis=0)

y_test_probs_xgb_3mer = np.concatenate(yprobs_3mers_xgb, axis=0)
y_test_probs_lgbm_3mer = np.concatenate(yprobs_3mers_lgbm, axis=0)
y_test_probs_cat_3mer = np.concatenate(yprobs_3mers_cat, axis=0)
# y_test_probs_svm_3mer = np.concatenate(yprobs_3mers_svm, axis=0)
y_test_probs_rf_3mer = np.concatenate(yprobs_3mers_rf, axis=0)


all_y_testss = np.concatenate(all_y_tests, axis=0)
all_y_preds = np.column_stack((y_test_preds_xgb_onehot, y_test_preds_lgbm_onehot, y_test_preds_cat_onehot, y_test_preds_svm_onehot, y_test_preds_rf_onehot,
                               y_test_preds_xgb_cksnap, y_test_preds_lgbm_cksnap, y_test_preds_cat_cksnap, y_test_preds_svm_cksnap, y_test_preds_rf_cksnap,
                               y_test_preds_xgb_3mer, y_test_preds_lgbm_3mer, y_test_preds_cat_3mer, y_test_preds_svm_3mer, y_test_preds_rf_3mer))

all_y_probs = np.column_stack((y_test_probs_xgb_onehot, y_test_probs_lgbm_onehot, y_test_probs_cat_onehot, y_test_probs_rf_onehot,
                               y_test_probs_xgb_cksnap, y_test_probs_lgbm_cksnap, y_test_probs_cat_cksnap, y_test_probs_rf_cksnap,
                               y_test_probs_xgb_3mer, y_test_probs_lgbm_3mer, y_test_probs_cat_3mer, y_test_probs_rf_3mer))


#%%
### Meta Classifier
from sklearn.linear_model import LogisticRegression
# meta_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# meta_classifier = XGBClassifier(random_state=42)
meta_classifier = LogisticRegression(random_state=0)

mean_ens_acc = []
mean_ens_mcc = []
mean_ens_sens = []
mean_ens_spe = []
mean_ens_f1 = []
mean_ens_pre = []
mean_ens_rec = []
mean_ens_auc = []

for i, (train_index_ens, test_index_ens) in enumerate(kf.split(all_y_preds, all_y_testss)): #all_y_probs
    X_train_enss, X_test_ens = all_y_preds[train_index_ens], all_y_preds[test_index_ens]
    y_train_enss, y_test_ens = all_y_testss[train_index_ens], all_y_testss[test_index_ens]
    
    X_train_ens, X_validation_ens, y_train_ens, y_validation_ens = train_test_split(X_train_enss, 
                    y_train_enss, test_size=0.01, random_state=92, shuffle=True)
    
    meta_classifier.fit(X_train_ens, y_train_ens)
    # meta_model = '/home/zeeshan/m5C_Mobeen/weights/danio_META_'+str(i+1)+'.pickle'
    # with open(meta_model, 'wb') as model_file:
    #     pickle.dump(meta_classifier, model_file)
        
    y_pred_ens = meta_classifier.predict(X_test_ens)
    acc_ens = metrics.accuracy_score(y_test_ens, y_pred_ens)
    print('Ensemble fold: ', i)
    mean_ens_acc.append(acc_ens)
    
    mcc_ens = metrics.matthews_corrcoef(y_test_ens, y_pred_ens)
    mean_ens_mcc.append(mcc_ens)
    
    
    confusion = confusion_matrix(y_test_ens, y_pred_ens)
    TN, FP, FN, TP = confusion.ravel()
    sensitivity = TP / float(TP + FN)
    mean_ens_sens.append(sensitivity)
    
    specificity = TN / float(TN + FP)
    mean_ens_spe.append(specificity)
    
    F1Score = (2 * TP) / float(2 * TP + FP + FN)
    mean_ens_f1.append(F1Score)
    
    precision = TP / float(TP + FP)
    mean_ens_pre.append(precision)
    
    recall = TP / float(TP + FN)
    mean_ens_rec.append(recall)
    
    #AUC
    y_pred_prob = meta_classifier.predict_proba(X_test_ens) #----
    y_probs = y_pred_prob[:,1]
    ROCArea = roc_auc_score(y_test_ens, y_probs)
    mean_ens_auc.append(ROCArea)
    
    
    ##########################
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42)

    # Apply t-SNE to your feature vectors
    X_tsne = tsne.fit_transform(X_test_ens)
    
    # Plot the t-SNE results with labels
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_pred_ens, cmap='viridis', marker='.')
    plt.colorbar()
    plt.savefig('tsne_plot_danio'+ str(i)+'.png', dpi=400) 
    # plt.title('t-SNE Plot of DNA Sequence Classification')
    # plt.xlabel('Dimension 1')
    # plt.ylabel('Dimension 2')
    plt.show()
    
    ##########################
    
    
    
mean_acc_ens = mean(mean_ens_acc)
mean_mcc_ens = mean(mean_ens_mcc)
mean_sens_ens = mean(mean_ens_sens)
mean_spe_ens = mean(mean_ens_spe)
mean_f1_ens = mean(mean_ens_f1)
mean_pre_ens = mean(mean_ens_pre)
mean_rec_ens = mean(mean_ens_rec)
mean_auc_ens = mean(mean_ens_auc)




print('Mean Mcc Ensemble: ', mean_mcc_ens)
print('Mean Acc Ensemble: ', mean_acc_ens)
print('Mean Sens Ensemble: ', mean_sens_ens)
print('Mean Spe Ensemble: ', mean_spe_ens)
print('Mean AUC Ensemble: ', mean_auc_ens)
print('Mean F1 Ensemble: ', mean_f1_ens)
print('Mean Precision Ensemble: ', mean_pre_ens)
print('Mean Recall Ensemble: ', mean_rec_ens)

    

   










