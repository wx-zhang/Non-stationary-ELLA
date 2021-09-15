import numpy as np
import sys
sys.path.append("./")
from problem.classification_problems import classical_syn_cls,classical_mnist,meta_syn_cls,meta_mnist
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from random import choice
sys.path.append("./")
from model.ELLA_non_stat import ELLA_non_stat
from model.ELLA import ELLA
from utils.data_process import prepare_syncls
from scipy.linalg import sqrtm, inv, norm, qr


# Choose which problem to run
problem = 'classification'
data_style = 'meta'



# Choose the classifier
logistic = False
ella = False
non_stat = True


# iteration parameter
total_iter = 200
count = 0
cur = 0

# Model parameter for all

seed = 2
feature_type = 'separate'
# Model parameter for ella and non-ella
writer = None
window = None
T = [20,10]
n_train = [100,20]

total_ella = []
total_non = []
total_log = []
x_axis = []





k=5
d=10
base_learner_kwargs =  {'max_iter':10000,  'solver':'liblinear', 'C': 1}

kwargs = {'window'      : window, 
          'seed'        : seed,
          'base_learner': LogisticRegression,
          'base_learner_kwargs':base_learner_kwargs  }

if problem == 'classification':
    kwargs['d'] = 10
    kwargs['k'] = 5
    if data_style == 'classical':
        model = classical_syn_cls
    if data_style == 'meta':
        model = meta_syn_cls
if problem == 'mnist':
    kwargs['angle'] = 5   
    if data_style == 'meta':
        model = meta_mnist
    if data_style == 'classical':
        model = classical_mnist



np.random.seed(seed)
Xs_train, Ys_binarized_train, Xs_test, Ys_binarized_test,_,_ = prepare_syncls(seed,k,d,T,n_train,method='meta')
acc = 0
print(f'{problem}, training size = {n_train}, window = {window}')    

grid_para = {'mu1': [1e-4, 5e-4, 1e-5, 5e-5,1e-6, 5e-6,1e-7],
             'mu3': [1e2,1e1,5e0,1e0,5e-1,1e-1,5e-2,1e-2,5e-3,1e-3, 1e-4, 1e-5,1e-6],
             'lam': [1e2,1e1,5e0,1e0,5e-1,1e-1,5e-2,1e-2,5e-3,1e-3, 1e-4, 1e-5,1e-6],
             'rhols': [1e0,5e-1,1e-1,5e-2, 1e-2,1e-3],
             'reg_weight': [6e-3, 1e-2,  6e-2, 3e-2,1e-1, 3e-1, 'avg']}



while count < total_iter:
    cur += 1
    # Do the iteration for different variables by changing x = cur

    kwargs['mu1'] = choice(grid_para['mu1'])
    kwargs['mu3'] = choice(grid_para['mu3'])
    kwargs['lam'] = choice(grid_para['lam'])
    kwargs['rhols'] = choice(grid_para['rhols'])
    kwargs['reg_weight'] = choice(grid_para['reg_weight'])

    model = ELLA_non_stat(**kwargs)
    cur_acc = []
    for i in range(sum(T)):
        ntrain = int(Xs_train[i].shape[0]*0.8)

        xtrain = Xs_train[i][:ntrain]
        ytrain = Ys_binarized_train[i][:ntrain]

        xdev = Xs_train[i][ntrain:]
        ydev = Ys_binarized_train[i][ntrain:]

        model.fit(xtrain,ytrain,i)
        cur_acc.append(model.score(xdev,ydev,i))
    cur_avg = np.mean(cur_acc[-10:])
    print (cur_avg)
    if cur_avg > acc:
        acc = cur_avg
        best_kwargs = kwargs
  
    count += 1
final_acc = []
model = ELLA_non_stat(**best_kwargs)
for i in range(sum(T)):
    model.fit(Xs_train, Ys_binarized_train,i)
    final_acc.append(model.score(Xs_test, Ys_binarized_test,i))
print (f'Fianl average accuracy is {np.mean(final_acc[-10:])}')

