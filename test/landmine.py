#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from sklearn.linear_model import LogisticRegression
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from scipy.io import loadmat


import sys
sys.path.append("./")
from ELLA_non_stat import ELLA_non_stat
from ELLA import ELLA
from data_process import multi_task_train_test_split, make_non_stat_data


data = loadmat('landminedata.mat')
Xs_lm = []
Ys_lm = []
T = data['feature'].shape[1]
for t in range(T):
    X_t = data['feature'][0,t]
    Xs_lm.append(np.hstack((X_t,np.ones((X_t.shape[0],1)))))
    Ys_lm.append(data['label'][0,t] == 1.0)
d = Xs_lm[0].shape[1]
k = 1



Xs_lm_train, Xs_lm_test, Ys_lm_train, Ys_lm_test = multi_task_train_test_split(Xs_lm,Ys_lm,train_size=0.5)
model1 = ELLA(d,k,LogisticRegression,{'C':10**0},mu=1,lam=10**-5)
for t in range(T):
    model1.fit(Xs_lm_train[t], Ys_lm_train[t], t)

model2 = ELLA_non_stat(d,k,LogisticRegression)
for t in range(T):
    model2.fit(Xs_lm_train[t], Ys_lm_train[t], t )

print ("Average AUC for ELLA:", np.mean([roc_auc_score(Ys_lm_test[t],
                                             model1.predict_logprobs(Xs_lm_test[t], t))
                               for t in range(1)]))
print ("Average AUC for non-stat-ELLA:", np.mean([roc_auc_score(Ys_lm_test[t],
                                             model2.predict_logprobs(Xs_lm_test[t], t))
                               for t in range(1)]))
acc1 = []
acc2 = []
for t in range(T):
    acc1.append(np.mean([model1.score(Xs_lm_test[i], Ys_lm_test[i], i) for i in range(t+1)]))
    acc2.append(np.mean([model2.score(Xs_lm_test[i], Ys_lm_test[i], i) for i in range(t+1)]))
plt.figure()
plt.title('Average accuracy for landmine problem')
plt.ylabel('Accuracy')
plt.xlabel('Tasks')
plt.plot(acc1,label = 'ELLA')
plt.plot(acc2,label = 'non-stat-ELLA')
plt.legend()
plt.show()

