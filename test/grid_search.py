import numpy as np
import sys
import random
import pandas as pd
from sklearn.linear_model import LogisticRegression
sys.path.append("./")
from utils.make_rotate_mnist_data import prepare_mnist

from utils.pca import PCA_feature
from model.NSELLA_GDCV import NSELLA_GDCV
from sklearn.model_selection import GridSearchCV
from utils.data_process import prepare_syncls

from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingGridSearchCV



seed = 1
random.seed(seed)
np.random.seed(seed)

k_mnist = 3
k_syncls = 5
kwargs = {}
kwargs['base_learner'] = LogisticRegression
kwargs['seed'] = seed
kwargs['base_learner_kwargs'] = {'max_iter':10000,  'solver':'liblinear', 'C': 1}
kwargs['max_iter'] = 50
problem = ['mnist', 'syncls']
train_size = [20, 50, 100]
window = [None, 10] 

def generate_mnist(n):
    T = 20
    angle = 5
    feature_type = 'separate'
    Xs_train_pca, ys_train, Xs_test_pca, ys_test, d = prepare_mnist(T,angle,n,seed,feature_type)
    Xs_train_pca = np.array(Xs_train_pca).transpose(1,0,2)
    Xs_test_pca = np.array(Xs_test_pca).transpose(1,0,2)
    ys_train = np.array(ys_train).transpose()
    ys_test = np.array(ys_test).transpose()
    return Xs_train_pca, ys_train, Xs_test_pca, ys_test, d

def generate_syncls(n,k,T):
    d = 10
    Xs_train, Ys_binarized_train, Xs_test, Ys_binarized_test,_,_ = prepare_syncls(seed,k,d,T,n)
    if isinstance(T,list):
        T = sum(T)
    Xs_train = np.array(Xs_train).transpose(1,0,2)
    Xs_test = np.array(Xs_test).transpose(1,0,2)
    Ys_binarized_train = np.array(Ys_binarized_train).transpose()
    Ys_binarized_test = np.array(Ys_binarized_test).transpose()
    return Xs_train, Ys_binarized_train, Xs_test, Ys_binarized_test,d 

# def train_model(kwargs,Xs_train, ys_train, Xs_test, ys_test):
#     for hyper in kwargs:
#         if hyper in  ['base_learner', 'base_learner_kwargs' , 'k', 'd']:
#             continue
#         print (hyper, '=', kwargs[hyper], end = ', ')
#     T = len(Xs_train)
#     model_non_ella = ELLA_non_stat(**kwargs)
#     acc = []
#     try:
#         for t in range(T):
#             model_non_ella.fit(Xs_train[t], ys_train[t], t)
#             acc.append(model_non_ella.score(Xs_test[t], ys_test[t], t))

#         print ('Accuracy = ', np.mean(acc))
#     except ValueError:
#         print ('Error')

 
# problem = ['mnist', 'syncls']
# mu1 = [1e-5, 1e-6, 1e-7]
# mu3 = [1e-2, 1e-4, 1e-6]
# lam = [5e-6, 1e-7, 5e-7, 1e-8, 5e-8]
# max_iter = [50, 100]
# rhols = [1e-1, 5e-2, 1e-2,5e-3,1e-3]
# window = [None, 10] 
# train_size = [20, 50, 100]
# reg_weight = [1e-3, 3e-3, 6e-3, 1e-2,  6e-2, 1e-1, 3e-1, 1,'avg']

grid_para = {'mu1': [1e-5, 1e-6, 1e-7],
             'mu3': [1e-2, 1e-4, 1e-6],
             'lam': [1e5,1e6,1e7,1e8],
             'rhols': [5e-2, 1e-2],
             'reg_weight': [6e-3, 1e-2,  6e-2, 1e-1, 3e-1, 'avg']}


# grid_para = {'mu1': [1e-5, 1e-6],
#              'mu3': [1e-2, 1e-4],
#              'lam': [5e-6, 1e-7],
#              'rhols': [1e-1, 5e-2],
#              'reg_weight': [ 3e-1,'avg']}




#problem = ['mnist']
# mu1 = [1e-05]
# mu3 = [1e-2]
# lam = [5e-07]
# max_iter = [100]
# rhols = [1e-3]
# window = [10] 
# train_size = [20]
# reg_method = [ 'weighted']
# reg_weight = [0.3]
p = 'syncls'
n = [100,20]
w = None
T = [20,10]
if p == 'mnist':
    Xs_train, ys_train, Xs_test, ys_test, d = generate_mnist(n)
    kwargs['k'] = k_mnist
if p == 'syncls':
    Xs_train, ys_train, Xs_test, ys_test, d = generate_syncls(n,k_syncls,T)
    kwargs['k'] = k_syncls
kwargs['d'] = d
kwargs['window'] = w
model = NSELLA_GDCV(**kwargs)


gscv = GridSearchCV(model, grid_para,n_jobs=15,verbose=3)
#gscv = HalvingGridSearchCV(model, grid_para,n_jobs=15,verbose=3)
gscv.fit(Xs_train, ys_train)
print (gscv.get_params())
data = gscv.cv_results_
df = pd.DataFrame(data)
df.to_excel(f'{p}_{n}_{w}_gscv_halving.xlsx')
print (gscv.score(Xs_test, ys_test))
# for p in problem:
#     print (f'Problem: {p}')
#     for n in train_size:
#         if p == 'mnist':
#             Xs_train, ys_train, Xs_test, ys_test, d = generate_mnist(n)
#             kwargs['k'] = k_mnist
#         if p == 'syncls':
#             Xs_train, ys_train, Xs_test, ys_test, d = generate_syncls(n,k_syncls)
#             kwargs['k'] = k_syncls
#         kwargs['d'] = d
#         for w in window:
#             kwargs['window'] = w
#             for m1 in mu1:
#                 kwargs['mu1'] = m1
#                 for m3 in mu3:
#                     kwargs['mu3'] = m3
#                     for l in lam:
#                         kwargs['lam'] = l
#                         for mi in max_iter:
#                             kwargs['max_iter'] = mi
#                             for rl in rhols:
#                                 kwargs['rhols'] = rl
#                                 for rm in reg_method:
#                                     kwargs['reg_method'] = rm
#                                     if rm == 'weighted':
#                                         for rw in reg_weight:
#                                             kwargs['reg_weight'] = rw
#                                             #try:
#                                             print ('train_size = ', n, end = ' ')
#                                             train_model(kwargs,Xs_train, ys_train, Xs_test, ys_test)
#                                             #except ValueError:
#                                             #    print ('ValueError')
#                                     else:
#                                         #try:
#                                         print ('train_size = ', n, end = ' ')
#                                         train_model(kwargs,Xs_train, ys_train, Xs_test, ys_test)
#                                         #except ValueError:
#                                         #    print ('ValueError')
