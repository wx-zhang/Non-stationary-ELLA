#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 11:09:18 2021

@author: wenxuanzhang
"""
from sklearn import decomposition
from sklearn.decomposition import KernelPCA
import numpy as np

def PCA_feature(Xs_train,Xs_test,ftype,d=None, seed = None):

    if d:
        pca = decomposition.PCA(n_components = d,random_state = seed)
    else:
        pca = decomposition.PCA(n_components = 0.95,random_state = seed)
    Xs_train_pca = []
    Xs_test_pca = []
    T = len(Xs_train)
    
    if ftype == 'separate':
        for i in range(T):
            pca.fit(Xs_train[i])           
            Xs_train_pca.append(pca.transform(Xs_train[i]))    
            Xs_test_pca.append(pca.transform(Xs_test[i]))
            if i == 0 and d == None:
                d = min(Xs_train_pca[0].shape[1],Xs_train_pca[-1].shape[1])
                pca = decomposition.PCA(n_components = d,random_state = seed)
                # pca =  KernelPCA(n_components=d,random_state = seed,kernel= 'poly')
                # pca.fit(Xs_train[0])           
                # Xs_train_pca[0]=(pca.transform(Xs_train[0]))    
                # Xs_test_pca[0]=(pca.transform(Xs_test[0]))



    
    if ftype == 'shared':
        nt = [Xs_train[i].shape[0] for i in range(T)]
        nt = [sum(nt[:i+1]) for i in range(T)]
        nt.insert(0,0)
        Xs = np.vstack(Xs_train)
        #print (Xs.shape)
        Xs_pca = pca.fit_transform(Xs)        
        for i in range(T):
            Xs_train_pca.append(Xs_pca[nt[i]:nt[i+1]])
            Xs_test_pca.append(pca.transform(Xs_test[i]))
            
    if ftype == 'first':
        pca.fit(Xs_train[0])
        for i in range(T):
            Xs_train_pca.append(pca.transform(Xs_train[i]))
            Xs_test_pca.append(pca.transform(Xs_test[i]))
    #print ([Xs_train_pca[i].shape for i in range(T)])
    return Xs_train_pca, Xs_test_pca 
        
        
        
    
    
    