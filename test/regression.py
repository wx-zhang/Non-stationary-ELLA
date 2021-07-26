from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.linalg import norm
import numpy as np
from scipy.stats import ortho_group
import matplotlib.pyplot as plt
import sys
sys.path.append("./")
from ELLA_non_stat import ELLA_non_stat
from ELLA import ELLA
from data_process import multi_task_train_test_split, make_non_stat_data

T = 100
d = 10
n = 100
k = 5
noise_var = .1

S_true = np.random.randn(k,T)
L_true = np.random.randn(d,k)
w_true = L_true.dot(S_true)


Xs = [np.hstack((np.random.randn(n,d-1), np.ones((n,1)))) for i in range(T)]
Ys = [Xs[i].dot(w_true[:,i]) + noise_var*np.random.randn(n,) for i in range(T)]
Xs = make_non_stat_data(d, T, Xs)
# break into train and test sets
Xs_train, Xs_test, Ys_train, Ys_test = multi_task_train_test_split(Xs,Ys,train_size=0.7)

# Regression Problem
model1 = ELLA(d,k,LinearRegression,mu=1,lam=10**-5)
for t in range(T):
    model1.fit(Xs_train[t], Ys_train[t], t)
#print ("ELLA, Average classification accuracy", np.mean([model1.score(Xs_test[t], Ys_test[t], t) for t in range(T)]))

model = ELLA_non_stat(d,k,LinearRegression)
for t in range(T):
    model.fit(Xs_train[t], Ys_train[t], t)
#     print (f" explained variance score at task {t}", model.score(Xs_test[t], Ys_test[t],t))
# # for t in range(T):
# #     print (f"Average explained variance score for task {t}", model.score(Xs_test[t], Ys_test[t],t))
# print ("Average explained variance score", np.mean([model.score(Xs_test[t], Ys_test[t], t) for t in range(T)]))

acc1 = []
acc2 = []
for t in range(T):
    acc1.append(np.mean([model1.score(Xs_test[i], Ys_test[i], i) for i in range(t+1)]))
    acc2.append(np.mean([model.score(Xs_test[i], Ys_test[i], i) for i in range(t+1)]))
plt.figure()
plt.title('Average accuracy for regression problem')
plt.ylabel('Explained Variance Score')
plt.xlabel('Tasks')
plt.plot(acc1,label = 'ELLA')
plt.plot(acc2,label = 'non-stat-ELLA')
plt.legend()
plt.show()










