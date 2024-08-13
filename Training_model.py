

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "2";


import optuna
from sklearn import tree
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from numpy import mean
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import random
from matplotlib import pyplot
from numpy import sort
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from sklearn.metrics import confusion_matrix, recall_score, roc_curve, roc_auc_score, auc
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_classif

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from catboost import CatBoostClassifier


import pandas as pd
import numpy as np
import re, os, sys

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

file=read_nucleotide_sequences(data_path) #ENAC encoded
cks = CKSNAP(file,gap=5) #gap=5 default
cc=np.array(cks)
data_only1 = cc[1:,2:]
data_only1 = data_only1.astype(np.float)




chemical_property = {
    'A': [1, 1, 1],
    'C': [0, 1, 0],
    'G': [1, 0, 0],
    'T': [0, 0, 1],
    'U': [0, 0, 1],
    '-': [0, 0, 0],
}

def NCP(fastas, **kw):

    AA = 'ACGT'
    encodings = []
    header = ['#', 'label']
    for i in range(1, len(fastas[0][1]) * 3 + 1):
        header.append('NCP.F'+str(i))
    encodings.append(header)

    for i in fastas:
        name, sequence, label = i[0], i[1], i[2]
        code = [name, label]
        for aa in sequence:
            code = code + chemical_property.get(aa, [0, 0, 0])
        encodings.append(code)
    return encodings

ncp=read_nucleotide_sequences(data_path)
enc=NCP(ncp)
dd=np.array(enc)
data_only2 = dd[1:,2:]
data_only2 = data_only2.astype(np.float)


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
    
data_io = dataProcessing(data_path,"fasta") #path,fileformat
data_only3 = data_io.reshape(len(data_io),41*4)

#------------------------------
# Trinucelotide (3-mer)
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

file=read_nucleotide_sequences(data_path)

tnc = TNC(file)
tnc = np.array(tnc)

data_only4 = tnc[1:,2:]
data_only4 = data_only4.astype(np.float)

#------------------------------

from collections import Counter

def TriNcleotideComposition(sequence, base):
    trincleotides = [nn1 + nn2 + nn3 for nn1 in base for nn2 in base for nn3 in base]
    tnc_dict = {}
    for triN in trincleotides:
        tnc_dict[triN] = 0
    for i in range(len(sequence) - 2):
        tnc_dict[sequence[i:i + 3]] += 1
    for key in tnc_dict:
       tnc_dict[key] /= (len(sequence) - 2)
    return tnc_dict

def PseEIIP(fastas, **kw):
    for i in fastas:
        if re.search('[^ACGT-]', i[1]):
            print('Error: illegal character included in the fasta sequences, only the "ACGT-" are allowed by this PseEIIP scheme.')
            return 0

    base = 'ACGT'

    EIIP_dict = {
        'A': 0.1260,
        'C': 0.1340,
        'G': 0.0806,
        'T': 0.1335,
    }

    trincleotides = [nn1 + nn2 + nn3 for nn1 in base for nn2 in base for nn3 in base]
    EIIPxyz = {}
    for triN in trincleotides:
        EIIPxyz[triN] = EIIP_dict[triN[0]] + EIIP_dict[triN[1]] + EIIP_dict[triN[2]]

    encodings = []
    header = ['#', 'label'] + trincleotides
    encodings.append(header)

    for i in fastas:
        name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
        code = [name, label]
        trincleotide_frequency = TriNcleotideComposition(sequence, base)
        code = code + [EIIPxyz[triN] * trincleotide_frequency[triN] for triN in trincleotides]
        encodings.append(code)
    return encodings


file=read_nucleotide_sequences(data_path)

pseeiip = PseEIIP(file)
pseeiip = np.array(pseeiip)

data_only6 = pseeiip[1:,2:]
data_only6 = data_only6.astype(np.float)
#---------------

def ENAC(fastas, window=5):
    
    kw = {'order': 'ACGT'}
    AA = kw['order'] if kw['order'] != None else 'ACGT'
    encodings = []
    header = ['#', 'label']
    for w in range(1, len(fastas[0][1]) - window + 2):
        for aa in AA:
            header.append('SW.' + str(w) + '.' + aa)
    encodings.append(header)

    for i in fastas:
        name, sequence, label = i[0], i[1], i[2]
        code = [name, label]
        for j in range(len(sequence)):
            if j < len(sequence) and j + window <= len(sequence):
                count = Counter(sequence[j:j + window])
                for key in count:
                    count[key] = count[key] / len(sequence[j:j + window])
                for aa in AA:
                    code.append(count[aa])
        encodings.append(code)
    return encodings

file=read_nucleotide_sequences(data_path)


enac = ENAC(file)
enac = np.array(enac)

data_only9 = enac[1:,2:]
data_only9 = data_only9.astype(np.float)

#-----------------------------------------------------------


# d = np.concatenate((data_only1,data_only2,data_only3,data_only6, data_only9),axis=1)
data_only=data_only4
length = len(data_only)



pos_lab = np.ones(int(length/2))
neg_lab = np.zeros(int(length/2))
labels = np.concatenate((pos_lab,neg_lab),axis=0)



c = list(zip(data_only, labels))
# random.shuffle(c)
random.Random(100).shuffle(c)
data_io, labels = zip(*c)
data_only=np.asarray(data_io)
labels=np.asarray(labels)

import shap
folds=5
scores = []

test_accs = []
test_mccs = []
sens = []
spec = []
aucs = []
f1 = []
prec = []
rec = []
aucprc=[]
   
kf = KFold(n_splits=folds, shuffle=True, random_state=4)

# kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=92)
for i, (train_index, test_index) in enumerate(kf.split(data_only,labels)):
    X_train0, X_test = data_only[train_index], data_only[test_index]
    y_train0, y_test = labels[train_index], labels[test_index]

    X_train, X_validation, y_train, y_validation = train_test_split(X_train0, y_train0, test_size=0.01, random_state=92, shuffle=True)
    
    ########################

    
    # xg = XGBClassifier() # ,objective='binary:logistic',eval_metric='logloss'
    
    # xgb_clf = xg.fit(X_train,y_train)
    # explainer = shap.TreeExplainer(xgb_clf) # , model_output = 'margin'
    # shap_values = explainer.shap_values(X_train)
    # importance = np.mean(abs(shap_values),axis=0)
    # zero_elements_indx = [i for i, v in enumerate(importance) if v <= 0.00]
    # # idx_sorted = np.argsort(importance)
    
    # X_traintest = pd.DataFrame(X_train)  ###-------- For simple importance calculation
    # X_testtest = pd.DataFrame(X_test)
    # selected_X_train = X_traintest.drop(zero_elements_indx,axis=1)
    # selected_X_test = X_testtest.drop(zero_elements_indx,axis=1)
    # selected_X_train = np.array(selected_X_train)
    # selected_X_test = np.array(selected_X_test)
    
    # shap.summary_plot(shap_values,X_train)
    # 
    #########################

       
    # xg = XGBClassifier(learning_rate=0.010474246167072487, max_depth=14, 
    #                 min_child_weight=1, gamma=0.6516423460350472, 
    #                 colsample_bytree=0.2464115690220173, n_estimators=833, seed=14) 
    
    # lgbm = lgb.LGBMClassifier(learning_rate=0.0753730090287657, max_depth=13,
    #             min_child_samples=11, min_child_weight=0.48660326897207984, min_split_gain=0.009254681493502266,
    #             n_estimators=357, num_leaves=49) 
    
    # svc = SVC(C=1, kernel='rbf', gamma='scale')
    
    # cbc = CatBoostClassifier(learning_rate=0.22322424314778602, max_depth=5,iterations=80,
    #                               boosting_type='Plain')
    
    rf = RandomForestClassifier(max_depth=30, bootstrap=False, max_features = 'log2',
                                    n_estimators=1400, min_samples_split=2,random_state=0)
    

    model2 = rf.fit(X_train,y_train)
    
    
    y_pred = model2.predict(X_test)
    
    test_acc = metrics.accuracy_score(y_test, y_pred)
    test_accs.append(test_acc)
    
    test_mcc = metrics.matthews_corrcoef(y_test, y_pred)
    test_mccs.append(test_mcc)
    
    
    confusion = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = confusion.ravel()
    sensitivity = TP / float(TP + FN)
    sens.append(sensitivity)
    
    specificity = TN / float(TN + FP)
    spec.append(specificity)
    
    F1Score = (2 * TP) / float(2 * TP + FP + FN)
    f1.append(F1Score)
    
    precision = TP / float(TP + FP)
    prec.append(precision)
    
    recall = TP / float(TP + FN)
    rec.append(recall)
    
    #AUC
    # y_pred_prob = model2.predict_proba(X_test) #----
    # y_probs = y_pred_prob[:,1]
    # ROCArea = roc_auc_score(y_test, y_probs)
    # aucs.append(ROCArea)
    
    from sklearn.metrics import precision_recall_curve
    precision2, recall2, _ = precision_recall_curve(y_test, y_pred)
    auc_prc = auc(recall2, precision2)
    aucprc.append(auc_prc)
    

    print('fold',i)
  
    # print(test_accs)
    i+=1
 
mean_acc = mean(test_accs)
mean_mcc = mean(test_mccs)
mean_sens = mean(sens)
mean_spec = mean(spec)
mean_f1 = mean(f1)
mean_prec = mean(prec)
# mean_auc = mean(aucs)
mean_rec = mean(rec)
mean_aucprc=mean(aucprc)
# mmm = mean(auc_newW)

print('***** Final Results *****')
print('Mean MCC: ', mean_mcc)
print('Mean Accuracy: ', mean_acc)
print('Mean Sensitivity: ', mean_sens)
print('Mean Specificity: ', mean_spec)
# print('Mean AUC: ', mean_auc)
print('Mean F1: ', mean_f1)
print('Mean Precision: ', mean_prec)
print('Mean Recall: ', mean_rec)
print('Mean PRcurve AUC: ',mean_aucprc)

