#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: wenxuanzhang
"""
import numpy as np

from args import parse_args
args = parse_args()
np.random.seed(args.seed)



if args.writer:
    import tensorflow as tf 
    import datetime
    date =  datetime.date.today().strftime("%y%m%d")
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f'results/logs_{args.dataset}/{date}/{current_time}'
    writer = tf.summary.create_file_writer(log_dir)
else:
    writer = None
    
    
    
# data
if args.dataset == 'synthetic':
    from utils.data_process import prepare_syncls
    Xs_train,ys_train,Xs_test,ys_test,true_p,true_L = prepare_syncls(args)
elif args.dataset == 'mnist':
    from utils.make_rotate_mnist_data import prepare_mnist
    Xs_train,ys_train,Xs_test,ys_test, args.d = prepare_mnist(args)
    true_p = None
    true_L = None
else: 
    raise NotImplementedError(f'Unknown dataset {args.dataset}')




# model
if args.base_learner == 'LR':
    from sklearn.linear_model import LogisticRegression
    baselearner = LogisticRegression
elif args.base_learner == 'Ridge':
    import sklearn.linear_model.Ridge as baselearner
else: 
    raise NotImplementedError(f'Unknown base learner {args.base_learner}')
    
    
base_learner_kwargs =  {'max_iter':10000,  'solver':'liblinear', 'C': 1, 'random_state': args.seed}
if 'NS' in args.baseline:
    from model.ELLA_non_stat import ELLA_non_stat
    model = ELLA_non_stat(args,baselearner, base_learner_kwargs,writer,true_p,true_L)
elif args.baseline == 'ELLA':
    from model.ELLA import ELLA
    model = ELLA(args.d,args.k,baselearner,base_learner_kwargs,mu = args.mu1,lam = args.mu3)
elif args.baseline == 'LR':
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(fit_intercept = False, **base_learner_kwargs)
else:
    raise NotImplementedError(f'Unknown baseline {args.baseline}')


acc = []
if args.data_style == 'classical':
    T = args.T
else:
    T = args.T + args.T_meta

for t in range(T):
    if args.baseline == 'LR':
        model.fit(Xs_train[t], ys_train[t])
        acc.append(model.score(Xs_test[t], ys_test[t]))
    else:
        model.fit(Xs_train[t], ys_train[t], t)
        acc.append(np.mean([model.score(Xs_test[i], ys_test[i],i) for i in range(t+1)]))
        
        
print (acc)