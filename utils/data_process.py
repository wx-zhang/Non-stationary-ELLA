#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.stats import ortho_group


def multi_task_train_test_split(Xs,Ys,train_size=0.5):
    Xs_train = []
    Ys_train = []
    Xs_test = []
    Ys_test = []
    for t in range(len(Xs)):
        X_train, X_test, y_train, y_test = train_test_split(Xs[t], np.squeeze(Ys[t]), train_size=train_size)
        Xs_train.append(X_train)
        Xs_test.append(X_test)
        Ys_train.append(y_train)
        Ys_test.append(y_test)
    return Xs_train, Xs_test, Ys_train, Ys_test


def make_non_stat_data(d,T,Xs,a):
    for i in range(T):
        Xs[i:] = [x@a for x in Xs[i:]]
    return Xs


def make_syncls(d,T,n_train,w_true,seed,train_split = 0.75 ,noise_var = .001):
    np.random.seed(seed)
    n = int(n_train // train_split)
    Xs = [np.hstack((np.random.randn(n,d-1), np.ones((n,1)))) for i in range(T)]
    # x = np.hstack((np.random.randn(n,d-1), np.ones((n,1))))
    # Xs = [x for i in range(T)]
    # y = Xw
    Ys = [Xs[i].dot(w_true[:,i]) + noise_var*np.random.randn(n,) for i in range(T)]
    Xs_train, Xs_test, Ys_train, Ys_test = multi_task_train_test_split(Xs,Ys,train_size=n_train)
    Ys_binarized_train = [Ys_train[i] > 0 for i in range(T)]
    Ys_binarized_test = [Ys_test[i] > 0 for i in range(T)]
    return Xs_train, Ys_binarized_train, Xs_test, Ys_binarized_test

def make_meta_syncls(d,Tl,n_trainl,w_true,seed):
    Xs_train1, Ys_binarized_train1, Xs_test1, Ys_binarized_test1 = make_syncls(d,Tl[0],n_trainl[0],w_true,seed)
    Xs_train2, Ys_binarized_train2, Xs_test2, Ys_binarized_test2 = make_syncls(d,Tl[1],n_trainl[1],w_true,seed+2)
    Xs_train = Xs_train1 + Xs_train2
    Ys_binarized_train = Ys_binarized_train1 + Ys_binarized_train2
    Xs_test = Xs_test1 + Xs_test2
    Ys_binarized_test = Ys_binarized_test1 + Ys_binarized_test2
    return Xs_train, Ys_binarized_train, Xs_test, Ys_binarized_test

def prepare_syncls(args): 
    if args.data_style == 'meta':
        T = args.T + args.T_meta
    else:
        T = args.T
    k = args.k
    d = args.d
    seed = args.seed
    S_true = np.random.randn(k,T)
    L_true = np.random.randn(d,k)
    w_true = L_true@(S_true)
    if args.data_style =='classical':
        n_train = args.n_train_classical
        Xs_train, Ys_binarized_train, Xs_test, Ys_binarized_test = make_syncls(d,T,n_train,w_true,seed)

    if args.data_style == 'meta':
        n_train_list = [args.n_train_classical, args.n_train_meta]
        Xs_train, Ys_binarized_train, Xs_test, Ys_binarized_test = make_meta_syncls(d,T,n_train_list,w_true,seed)

    size = 2
    a,_ = np.float32(ortho_group.rvs(size=size, dim=d,random_state=seed))
    Xs_train = make_non_stat_data(d, T, Xs_train,a)
    Xs_test = make_non_stat_data(d, T, Xs_test,a)

    return Xs_train, Ys_binarized_train, Xs_test, Ys_binarized_test,a.T,L_true
