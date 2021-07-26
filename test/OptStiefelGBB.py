import numpy as np
from scipy.linalg import norm, qr
import sys
sys.path.append("../")
from utils.ortho_opt import ortho_opt

np.random.seed(1)
n = 10
D = np.random.randn(n,n)
D = D@D.T
a = np.random.randn(n,1)
s = np.random.randn(n,1)
def h(X):
    return (a - X@s).T @ D @ (a - X@s) + norm(s,ord=1)
def h_grad(X):
    return  -2 * D @ (a - X @ s) @ (s.T)




X = np.identity(n)
method = 'curvilinear'
print (h(X))
X = ortho_opt(X,h,h_grad,method)
print (h(X))



