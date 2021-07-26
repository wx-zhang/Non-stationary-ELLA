from sklearn.model_selection  import train_test_split
import numpy as np
from scipy.stats import ortho_group
import matplotlib.pyplot as plt
def multi_task_train_test_split(Xs,Ys,train_size=0.8):
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
import sys
sys.path.append("./")
from ELLA import ELLA
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression
from scipy.linalg import norm
import numpy as np

T = 100
d = 10
n = 100
k = 5
noise_var = .001

model = ELLA(d,k,Ridge,mu=1,lam=10**-5)

S_true = np.random.randn(k,T)
L_true = np.random.randn(d,k)
w_true = L_true.dot(S_true)

'''
# make sure to add a bias term (it is not done automatically)
Xs = [np.hstack((np.random.randn(n,d-1), np.ones((n,1)))) for i in range(T)]
# generate the synthetic labels
Ys = [Xs[i].dot(w_true[:,i]) + noise_var*np.random.randn(n,) for i in range(T)]
'''
seed = 1
size = 2
a,_ = np.float32(ortho_group.rvs(size=size, dim=d, random_state=seed))
#a = np.identity(d)
# generate x,y
Xs = [np.hstack((np.random.randn(n,d-1), np.ones((n,1)))) for i in range(T)]
Ys = [Xs[i].dot(w_true[:,i]) + noise_var*np.random.randn(n,) for i in range(T)]
for i in range(T-1):
    Xs[i+1:] = [x@a for x in Xs[i+1:]]

# break into train and test sets
Xs_train, Xs_test, Ys_train, Ys_test = multi_task_train_test_split(Xs,Ys,train_size=0.7)

for t in range(T):
    model.fit(Xs_train[t], Ys_train[t], t)
print ("Average explained variance score", np.mean([model.score(Xs_test[t], Ys_test[t], t) for t in range(T)]))

# Try out a classification problem
Ys_binarized_train = [Ys_train[i] > 0 for i in range(T)]
Ys_binarized_test = [Ys_test[i] > 0 for i in range(T)]

model = ELLA(d,k,LogisticRegression,mu=1,lam=10**-5)
tr = []
te = []
for t in range(T):
    model.fit(Xs_train[t], Ys_binarized_train[t], t)
    tr.append(model.score(Xs_train[t], Ys_binarized_train[t], t))
    te.append(model.score(Xs_test[t], Ys_binarized_test[t], t))
    print (f"Average classification accuracy at task {t}", np.mean([model.score(Xs_test[i], Ys_binarized_test[i], i) for i in range(t+1)]))
plt.plot(range(T),tr,range(T),te)
plt.plot( [np.mean([model.score(Xs_test[i], Ys_binarized_test[i], i) for i in range(t+1)])for t in range(T)])
print ("Average classification accuracy", np.mean([model.score(Xs_test[t], Ys_binarized_test[t], t) for t in range(T)]))