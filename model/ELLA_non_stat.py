""" Alpha version of a version of ELLA that plays nicely with sklearn
	@author: Paul Ruvolo
"""


#import numpy as np
import autograd.numpy as np 
from scipy.special import logsumexp
from scipy.linalg import sqrtm, inv, norm, qr,solve
import sklearn
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression, Lasso
from sklearn.metrics import accuracy_score, explained_variance_score

import tensorflow as tf



import sys
sys.path.append("./")
from utils.ortho_opt import ortho_opt
import pymanopt
from pymanopt.manifolds import Stiefel
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent
from pymanopt.solvers import ConjugateGradient
from pymanopt.solvers import TrustRegions


class ELLA_non_stat(object):
	""" The ELLA model """
	def __init__(self, d, k, base_learner, base_learner_kwargs = {}, 
				 mu1 = 1e-3,
				 mu3= 1e-3, 
				 lam = 1e1, 
				 max_iter = 100,
				 rhols = 1e-5,
				 window = None,
				 reg_weight = 1,
				 k_init =True,
				 seed = 1,
				 writer = None,
				 true_psi = None,
				 method = 'QR',
				 true_L = None,
				 true_s = None):
		""" Initializes a new model for the given base_learner.
			mu1: L1 penalty on s for sparsity

			lam: Regularization term on psi
			d: the number of parameters for the base learner
			k: the number of latent model components
			max_iter: Maximum iteration for the optimization of psi
			base_learner: the base learner to use (currently can only be
				LinearRegression, Ridge, or LogisticRegression).
			base_learner_kwargs: keyword arguments to base learner (for instance to
								 specify regularization strength)
			NOTE: currently only binary logistic regression is supported
		"""

			
		np.random.seed(seed)
		self.d = d
		self.k = k
		self.L = np.matrix(np.random.randn(d,k))
		self.L_list = []
		self.psi = np.matrix(np.identity(d))
		self.psi_list = []
		self.psig = [np.matrix(np.identity(self.d))]
		self.D_list = []
		self.alpha_list = []
		self.s_list = []
		self.T = 0
		self.mu1 = mu1
		self.mu3 = mu3
		self.lam = lam
		self.rhols = rhols
		self.max_iter = max_iter
		self.k_init = k_init
		self.window = window
		self.reg_weight = reg_weight
		self.reg_psi = np.zeros_like(self.psi)
		self.writer = writer
		self.true_psi = true_psi
		self.true_L = true_L
		self.psimethod = method
		self.true_s = true_s


		print (d,k,self.mu1,self.mu3,self.lam, self.max_iter, self.rhols, self.reg_weight)

		if base_learner in [LinearRegression, Ridge]:
			self.perf_metric = explained_variance_score
		elif base_learner in [LogisticRegression]:
			self.perf_metric = accuracy_score
		else:
			raise Exception("Unsupported Base Learner")

		self.base_learner = base_learner
		self.base_learner_kwargs = base_learner_kwargs
		self.base_learner_kwargs['random_state'] = seed


		# self.L = self.true_L
		# self.psi = self.true_psi

	def fit(self, X, y, task_id):
		""" Fit the model to a new batch of training data.  The task_id must
			start at 0 and increase by one each time this function is called.
			Currently you cannot add new data to old tasks.
			X: the training data
			y: the trianing labels
			task_id: the id of the task
		"""


		# init for every task
		self.T += 1
		self.psig.append(np.matrix(np.identity(self.d)))

		true_psig = np.identity(self.d)
		for i in range(self.T):
			true_psig = true_psig @ self.true_psi



		# alpha, D 
		single_task_model = self.base_learner(fit_intercept = False, **self.base_learner_kwargs).fit(X, y)
		alpha_t = np.matrix(single_task_model.coef_).T
		#print ('alpha diff', norm(true_psig@self.true_L @ self.true_s[:,task_id] - single_task_model.coef_))
		D_t = self.get_hessian(single_task_model, X, y)       
		D_t_sqrt = np.matrix(sqrtm(D_t))
		
		self.D_list.append(D_t)
		self.alpha_list.append(alpha_t)
		

		# s       
		sparse_encode = Lasso(alpha = self.mu1 / (X.shape[0] * 2.0),
							  fit_intercept = False, tol=1e-10).fit(D_t_sqrt@self.psi@self.psig[self.T-1]@self.L,
														 D_t_sqrt@alpha_t)
		if self.k_init and task_id < self.k:
			
			# set Ls = L[:,T] to init L with theta
			self.init_train()           
		else:
			#print (np.matrix(sparse_encode.coef_).T)
			sparse_coeffs = np.matrix(sparse_encode.coef_).T

			self.s_list.append(sparse_coeffs)  
			#print ('S diff',norm(self.true_s[:,task_id] - sparse_coeffs))
	
			
			# update psi,L
			self.updateTransionMatrix()
			self.updateLatentSpace()
			
			


			

			
		  
			self.revive_dead_components()
		
		
	def updateLatentSpace(self):
		if self.T < 20:
			# null the gradient
			C = np.matrix(np.zeros_like(self.L))
			D = np.matrix(np.zeros((self.k*self.d,self.k*self.d)))

			for i in range(self.T):
				C += self.psig[i+1].T @ self.D_list[i] @ self.alpha_list[i] @ self.s_list[i].T
				A = np.matrix(self.psig[i+1].T @ self.D_list[i] @ self.psig[i+1])
				B = np.matrix(self.s_list[i] @ self.s_list[i].T)
				D += np.kron(B.T,A)
				D += self.mu3 * np.identity(self.d*self.k)
				
			C_vectorized = C.reshape((C.size,1),order='F')
			L_vectorized = inv(D) @ C_vectorized 
			self.L = L_vectorized.reshape((self.d, self.k),order='F')
		
		#self.L = self.true_L

		self.L_list.append(self.L)
		if self.writer:
			with self.writer.as_default():
				self.T >1 and tf.summary.scalar('L_diff',norm(self.L - self.L_list[-2]),step=self.T) 

	
	def updateTransionMatrix(self):  
		# if self.T > self.k: 
		# 	while (1) : 
		# 		prop =   self.regularization_psi(self.psi)/self.loss(self.psi,self.L)
		# 		if prop > self.psi_prop*1.2:
		# 			self.lam = 0.9*self.lam
		# 		elif prop < self.psi_prop*0.2:
		# 			self.lam = 1.1*self.lam
		# 		else:
		# 			break


		# if self.T ==6:
		# 	self.reg_psi = np.mean(self.psi_list[1:])
		if self.T == 1:
			self.reg_psi = self.true_psi

		# if self.T == 6:
		# 	#print (np.array(self.alpha_list[:-1])[:,:,0].shape)
		# 	A = np.array(self.alpha_list[:-1])[:,:,0].T
		# 	B = np.array(self.alpha_list[1:])[:,:,0].T
		# 	self.reg_psi = 0.5* B @A.T@inv(A@A.T) + np.mean(self.psi_list)
		if self.T < 100:
			# self.psi =  ortho_opt(self.psi,self.h,self.g,self.psimethod,writer = self.writer)
			# self.psi = self.true_psi
			@pymanopt.function.Autograd
			def cost(X):
				h = 0
				if self.window and self.T > self.window:
					start = self.T - self.window
				else:
					start = 0
				for i in range(start,self.T):
					psig = np.array(self.psig[i])
					L = np.array(self.L)
					D = np.array(self.D_list[i])
					s  = np.array(self.s_list[i])
					a = np.array(self.alpha_list[i])
					h += 1/self.T * (self.ell(L, 
											  X@psig, 
											  s,
											  a, 
											  D))
				reg = np.array(self.reg_psi)
				h += self.lam * np.sum((X - reg)**2)
				return np.sum(h)

			manifold = Stiefel(self.d,self.d)
			problem = Problem(manifold, cost,verbosity=0)
			solver = ConjugateGradient()
			self.psi = solver.solve(problem)

		#print ('train_loss', self.h(self.psi),end='  ')
		
		
		if self.writer:
			with self.writer.as_default():
				if self.T >1:
					tf.summary.scalar('psi_diff',norm(self.psi - self.psi_list[-1]), step = self.T)   
				#tf.summary.scalar('psireg',self.regularization_psi(self.psi), step = self.T) 
				tf.summary.scalar('lreg',self.regularization_l(self.L), step = self.T) 
				tf.summary.scalar('ell',self.ell(self.L,self.psi,self.s_list[-1],self.alpha_list[-1],self.D_list[-1]), step = self.T) 


		#self.psi = self.true_psi

		self.psi_list.append(self.psi)
		self.updateGroupTransitionMatrix()



		
		# else:

		if self.T >6:
			if self.reg_weight  == 'avg':
				self.reg_psi = sum(self.psi_list) / self.T
			elif isinstance(self.reg_weight,float) or isinstance(self.reg_weight,int) :
				self.reg_psi = self.reg_weight * self.psi + (1-self.reg_weight) * self.reg_psi

		# for i in range(self.T):
		# 	print (self.T, i, (self.ell(self.L, 
		# 							  self.psig[i+1], 
		# 							  self.s_list[i],
		# 							  self.alpha_list[i], 
		# 							  self.D_list[i])))
	   

	def updateGroupTransitionMatrix(self):
		# for i in range(1,self.T+1):
		# 	self.psig[-1] = self.psig[-1] @ self.psi_list[-i]

		# self.psig[-1] = self.psi @ self.psig[-2]
		# true_psig = np.identity(self.d)
		# for i in range(self.T):
		# 	true_psig = true_psig @ self.true_psi

		# for i in range(1,self.T+1):
		# 	self.psig[-1] = self.psig[-1] @ self.psi

		for j in range(1,self.T+1):
			self.psig[-j] = self.psi @self.psig[-j-1]
			#self.psig[i] = self.psig[i]@self.psi_list[-1] 


	def loss(self,psi,L):
		return self.h(psi)+ self.mu3 * norm((L),ord = 'fro')+ self.mu1*norm(self.s_list[self.T-1],ord=1)

	def ell(self, L, psi, s, a, D):
		return np.sum((a - psi@L@s).T @ D @ (a - psi@L@s) + norm(s,ord=1))

	def regularization_psi(self,X):

		return norm(X - self.reg_psi)**2


	def regularization_l(self,X):
		return norm(X,ord='fro')**2

	def h(self, X):
		h = 0
		if self.window and self.T > self.window:
			start = self.T - self.window
		else:
			start = 0



		for i in range(start,self.T):
			h += 1/self.T * (self.ell(self.L, 
									  X@self.psig[i], 
									  self.s_list[i],
									  self.alpha_list[i], 
									  self.D_list[i]))



		h += self.lam * self.regularization_psi(X)
		return np.sum(h)

	def g(self,X):
		G = np.matrix(np.zeros((self.d,self.d)))
		if self.window and self.T > self.window:
			start = self.T - self.window
		else:
			start = 0

		for i in range(start, self.T):
			b = np.matrix(self.psig[i] @ self.L @ self.s_list[i])
			G += 1/self.T*( -2 * self.D_list[i] @ \
				 (self.alpha_list[i] - X @ b) @ (b.T))

		G += 2 * self.lam * (X - self.reg_psi)


		return G     
		
		
	def get_hessian(self, model, X, y):
		""" ELLA requires that each single task learner provide the Hessian
			of the loss function evaluated around the optimal single task
			parameters.  This funciton implements this for the base learners
			that are currently supported """
		alpha_t = model.coef_
		if self.base_learner == LinearRegression:
			return X.T.dot(X)/(2.0 * X.shape[0])
		elif self.base_learner == Ridge:
			return X.T.dot(X)/(2.0 * X.shape[0]) + model.alpha * np.eye(self.d, self.d)
		elif self.base_learner == LogisticRegression:
			preds = 1. / (1.0 + np.exp(-X.dot(alpha_t.T)))
			base = np.tile(preds * (1 - preds), (1, X.shape[1]))
			hessian = (np.multiply(X, base)).T.dot(X) / (2.0 * X.shape[0])
			return np.matrix(hessian + np.eye(self.d,self.d) / (2.0 * model.C))
	
	def init_train(self):
		s = np.zeros((self.k,1))
		s[self.T-1] = 1.0
		self.s_list.append(s)

		# self.psi = np.identity(self.d)
		# self.psi_list.append(self.psi)
		# self.updateGroupTransitionMatrix()
		self.updateTransionMatrix()

		self.L[:,self.T-1] = self.psig[self.T].T @ self.alpha_list[-1]    
		self.L_list.append(self.L)


	

	
				
		
	def predict(self, X, task_id):
		""" Output ELLA's predictions for the specified data on the specified
			task_id.  If using a continuous model (Ridge and LinearRegression)
			the result is the prediction.  If using a classification model
			(LogisticRgerssion) the output is currently a probability.
		"""
		if self.base_learner == LinearRegression or self.base_learner == Ridge:
			return (X@self.psig[task_id+1]@self.L@self.s_list[task_id])
		elif self.base_learner == LogisticRegression:

			return 1. / (1.0 + np.exp(-(X@self.psig[task_id+1]@self.L@self.s_list[task_id]))) > 0.5
	

	def predict_probs(self, X, task_id):
		""" Output ELLA's predictions for the specified data on the specified
			task_id.  If using a continuous model (Ridge and LinearRegression)
			the result is the prediction.  If using a classification model
			(LogisticRgerssion) the output is currently a probability.
		"""
		if self.base_learner == LinearRegression or self.base_learner == Ridge:
			raise Exception("This base learner does not support predicting probabilities")
		elif self.base_learner == LogisticRegression:
			return np.exp(self.predict_logprobs(X, task_id))

	def predict_logprobs(self, X, task_id):
		""" Output ELLA's predictions for the specified data on the specified
			task_id.  If using a continuous model (Ridge and LinearRegression)
			the result is the prediction.  If using a classification model
			(LogisticRgerssion) the output is currently a probability.
		"""
		if self.base_learner == LinearRegression or self.base_learner == Ridge:
			raise Exception("This base learner does not support predicting probabilities")
		elif self.base_learner == LogisticRegression:
			return -logsumexp(np.hstack((np.zeros((X.shape[0], 1)), -X@self.psig[task_id+1]@self.L@self.s_list[task_id])), axis = 1)

	def revive_dead_components(self):
			""" re-initailizes any components that have decayed to 0 """
			flag = 0
			for i,val in enumerate(np.sum(np.array(self.L), axis = 0)):
				if abs(val) < 10 ** -8:
					flag += 1
					self.L[:, i] = self.psig[-flag].T @ self.alpha_list[-flag]
					self.s_list[-flag] = np.zeros((self.k,1))
					self.s_list[i] = 1.0
			self.L = np.matrix(self.L)    

			
	def score(self, X, y, task_id):
			""" Output the score for ELLA's model on the specified testing data.
				If using a continuous model (Ridge and LinearRegression)
				the score is explained variance.  If using a classification model
				(LogisticRegression) the score is accuracy.
			"""

			return self.perf_metric(self.predict(X, task_id), y)









		