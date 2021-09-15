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





import sys
sys.path.append("./")
from utils.ortho_opt import ortho_opt
import pymanopt
from pymanopt.manifolds import Stiefel
from pymanopt import Problem
from pymanopt.solvers import ConjugateGradient



class ELLA_non_stat(object):
	""" The ELLA model """
	def __init__(self,args,baselearner,base_learner_kwargs,writer=None, true_p=None,true_L=None):


		self.d = args.d
		self.k = args.k
		self.mu1 = args.mu1
		self.mu3 = args.mu3
		self.mu2 = args.mu2
		self.reg_weight = args.reg_weight
		self.reg_method = args.reg_method
		
		self.k_init = args.k_init
		self.multi_init = args.multi_init
		self.n_multi_init = args.n_multi_init
		self.init_epoch = args.init_epoch
		
		self.base_learner = baselearner
		self.base_learner_kwargs = base_learner_kwargs
		self.stop_update_L = args.stop_update_L
		self.stop_update_p = args.stop_update_p
		self.use_true_p = args.true_p
		self.use_true_pavg = args.true_pavg
		self.use_true_L = args.true_L
		
		self.rhols = args.rhols
		self.max_iter = args.max_iter		
		self.window = args.window		
		self.opt_method = args.opt_method
		
		
		self.writer = writer
		self.true_p = true_p
		self.true_L = true_L
		
		if self.writer:
			import tensorflow as tf

			
		if self.base_learner in [LinearRegression, Ridge]:
			self.perf_metric = explained_variance_score
		elif self.base_learner in [LogisticRegression]:
			self.perf_metric = accuracy_score
		else:
			raise Exception("Unsupported Base Learner")
		
		
		self.L = np.matrix(np.random.randn(self.d,self.k))
		self.L_list = []
		self.psi = np.matrix(np.identity(self.d))
		self.psi_list = []
		self.psig = [np.matrix(np.identity(self.d))]
		self.D_list = []
		self.alpha_list = []
		self.s_list = []
		self.T = 0
		self.init_buffer = {}
		self.reg_psi = self.psi

		
		if self.use_true_pavg:
			self.reg_psi = self.true_p
			self.multi_init = False

		if self.use_true_p:
			self.k_init = False
			self.L = self.true_L

		if self.use_true_p:
			self.psi = self.true_p
			self.multi_init = False
			self.reg_psi = self.psi
		
		print (self.d,self.k,self.k_init,self.multi_init,self.n_multi_init)
		


	def fit(self, X, y, task_id):
		""" Fit the model to a new batch of training data.  The task_id must
			start at 0 and increase by one each time this function is called.
			Currently you cannot add new data to old tasks.
			X: the training data
			y: the trianing labels
			task_id: the id of the task
		"""
		self.T += 1
		self.L_list.append(self.L)
		self.psi_list.append(self.psi)
		self.s_list.append(np.zeros((self.k,1)))
		self.psig.append(self.psi)


		# Get alpha, D for every task
		alpha_t, D_t = self.get_D_alpha(X, y)
		self.D_list.append(D_t)
		self.alpha_list.append(alpha_t)
		
		# Initlization
		if self.multi_init and task_id < self.n_multi_init:
			# init psi_avg
			self.init_buffer[task_id] = [X,y]
			if task_id == self.n_multi_init-1:
				self.multi_task_init(self.init_buffer)


				
		elif self.k_init and task_id < self.k:
			self.k_L_init(task_id)

		else:  
			self.three_way_optimization(X,y,task_id)

		self.tf_log('fit')

				
		
		
	def three_way_optimization(self,X,y,task_id):
		self.s_list[task_id] =  self.get_sparse_coeff(X,task_id)


		self.updateTransionMatrix(task_id)
		self.psi_list[task_id] = self.psi
		self.updateGroupTransitionMatrix(task_id)
			
		self.updateLatentSpace(task_id)
		self.L_list[task_id]=self.L 
		


	def get_D_alpha(self, X, y):
		single_task_model = self.base_learner(fit_intercept = False, **self.base_learner_kwargs).fit(X, y)
		alpha_t = np.matrix(single_task_model.coef_).T
		D_t = self.get_hessian(single_task_model, X, y)       
		return alpha_t, D_t

	def get_sparse_coeff(self,X,t):
		D_t_sqrt = np.matrix(sqrtm(self.D_list[t]))
		sparse_encode = Lasso(alpha = self.mu1 / (X.shape[0] * 2.0),
							  fit_intercept = False, tol=1e-10).fit(D_t_sqrt@self.psi@self.psig[t]@self.L,
														 D_t_sqrt@self.alpha_list[t])
		sparse_coeffs = np.matrix(sparse_encode.coef_).T
		#print (sparse_encode.coef_)
		return sparse_coeffs

	def updateLatentSpace(self,t):
		# t currnet task id

		if self.use_true_L:
			self.L = self.true_L
			return 

		if t < self.stop_update_L :
			# null the gradient
			C = np.matrix(np.zeros_like(self.L))
			D = np.matrix(np.zeros((self.k*self.d,self.k*self.d)))

			for i in range(t+1):
				C += self.psig[i+1].T @ self.D_list[i] @ self.alpha_list[i] @ self.s_list[i].T
				A = np.matrix(self.psig[i+1].T @ self.D_list[i] @ self.psig[i+1])
				B = np.matrix(self.s_list[i] @ self.s_list[i].T)
				D += np.kron(B.T,A)
				D += self.mu3 * np.identity(self.d*self.k)
				
			C_vectorized = C.reshape((C.size,1),order='F')
			L_vectorized = inv(D) @ C_vectorized 
			self.L = L_vectorized.reshape((self.d, self.k),order='F')
			
			self.revive_dead_components()


	
	def updateTransionMatrix(self,t): 

		if self.use_true_p:
			self.psi = self.true_p
			return 

		if t < self.stop_update_p:
			if self.opt_method == 'pymanopt':

				@pymanopt.function.Autograd
				def cost(X):
					h = 0
					if self.window and t > self.window:
						start = t + 1 - self.window
					else:
						start = 0
					for i in range(start,t+1):
						psig = np.array(self.psig[i])
						L = np.array(self.L)
						D = np.array(self.D_list[i])
						s  = np.array(self.s_list[i])
						a = np.array(self.alpha_list[i])
						h += 1/(t+1-start) * (self.ell(L, 
												  X@psig, 
												  s,
												  a, 
												  D))
					reg = np.array(self.reg_psi)
					h += self.mu2 * np.sum((X - reg)**2)
					return np.sum(h)

				manifold = Stiefel(self.d,self.d)
				problem = Problem(manifold, cost,verbosity=0)
				solver = ConjugateGradient()
				self.psi = solver.solve(problem)

			else:
				self.psi = ortho_opt(self.psi,self.h,self.g,self.opt_method,writer = self.writer)



			if self.reg_method  == 'avg':
				self.reg_psi = sum(self.psi_list) / self.T
			elif self.reg_method == 'ewma' :
				self.reg_psi = (1-self.reg_weight) * self.psi + self.reg_weight * self.reg_psi
			else:
				raise NotImplementedError(f'Unknown update method {self.reg_method}')


	def updateGroupTransitionMatrix(self,t):	

		self.psig[t+1] = self.psi @ self.psig[t]


	def loss(self,psi,L):
		return self.h(psi)+ self.mu3 * norm((L),ord = 'fro')+ self.mu1*np.sum([norm(self.s_list[i],ord=1) for i in self.T])

	def ell(self, L, psi, s, a, D,sreg=True):
		ell =  np.sum((a - psi@L@s).T @ D @ (a - psi@L@s))
		if sreg:
			ell += norm(s,ord=1)

		return ell
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
			h += 1/(self.T-start) * (self.ell(self.L, 
									  X@self.psig[i], 
									  self.s_list[i],
									  self.alpha_list[i], 
									  self.D_list[i]))



		h += self.mu2 * self.regularization_psi(X)
		return np.sum(h)

	def g(self,X):
		G = np.matrix(np.zeros((self.d,self.d)))
		if self.window and self.T > self.window:
			start = self.T - self.window
		else:
			start = 0

		for i in range(start, self.T):
			b = np.matrix(self.psig[i] @ self.L @ self.s_list[i])
			G += 1/(self.T-start)*( -2 * self.D_list[i] @ \
				 (self.alpha_list[i] - X @ b) @ (b.T))

		G += 2 * self.mu2 * (X - self.reg_psi)


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
	
	def k_L_init(self,t):
		s = np.zeros((self.k,1))
		s[t] = 1.0
		self.s_list[t] = s
		
		self.updateTransionMatrix(t)
		self.psi_list[t] = self.psi
		self.updateGroupTransitionMatrix(t)

		self.L[:,t] = self.psig[t+1].T @ self.alpha_list[t] 
		self.L_list[t] = self.L


	def multi_task_init(self,init_buffer):
		
		if self.n_multi_init < self.k:
			raise ValueError('Suggest to use at least k components for the multi-task init')
		
		if self.k_init:
			# initilize L
			for i in range(self.k):
				self.k_L_init(i)


		self.log_step_count = 0
		for epoch in range(self.init_epoch):
			for i in range(self.n_multi_init):
				task_id = i
				X = init_buffer[task_id][0]
				y = init_buffer[task_id][1]
				self.s_list[task_id] =  self.get_sparse_coeff(X,task_id) 

			@pymanopt.function.Autograd
			def cost(X):
				h = 0
				for i in range(self.n_multi_init):
					psig = X
					for j in range(i):
						psig = psig@X
					L = np.array(self.L)
					D = np.array(self.D_list[i])
					s  = np.array(self.s_list[i])
					a = np.array(self.alpha_list[i])
					h +=  (self.ell(L, psig, s,a,D,sreg=False))
				return np.sum(h)

			manifold = Stiefel(self.d,self.d)
			problem = Problem(manifold, cost,verbosity=0)
			solver = ConjugateGradient(minstepsize=1e-10)
			self.psi = solver.solve(problem,x=np.array(self.psi))


			self.psi_list[-1] = self.psi
			for task_id in range(self.n_multi_init):
				self.updateGroupTransitionMatrix(task_id)	
				self.s_list[task_id] =  self.get_sparse_coeff(init_buffer[task_id][0],task_id)
			self.updateLatentSpace(self.n_multi_init-1)
			self.L_list[task_id]=self.L 


			self.tf_log('multi-init')
			self.log_step_count += 1
		self.reg_psi = self.psi
		
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
	
	def tf_log(self, process):
		if self.writer:
			with self.writer.as_default():
				if process == 'multi-init':
					self.true_p and tf.summary.scalar(f'{process}_psi_diff_true_init',norm(self.psi - self.true_p), step = self.log_step_count) 
					self.true_L and tf.summary.scalar(f'{process}_L_diff_true_init',norm(self.L - self.true_L), step = self.log_step_count) 

				if process == 'fit':
					self.T >1   and	tf.summary.scalar('psi_diff',norm(self.psi - self.psi_list[-1]), step = self.T)   
					self.use_true_p and tf.summary.scalar('psi_diff_true',norm(self.psi - self.true_p), step = self.T) 
					self.use_true_p and tf.summary.scalar('psi_avg_diff_true', norm(self.reg_psi -self.true_p ), step = self.T)

					self.T >1 and tf.summary.scalar('L_diff',norm(self.L - self.L_list[-2]),step=self.T) 
					self.use_true_L and tf.summary.scalar('L_diff_true',norm(self.L - self.true_L),step=self.T) 

					tf.summary.scalar('psireg',self.regularization_psi(self.psi), step = self.T) 
					tf.summary.scalar('lreg',self.regularization_l(self.L), step = self.T) 
					tf.summary.scalar('ell',self.ell(self.L,self.psig[-1],self.s_list[-1],self.alpha_list[-1],self.D_list[-1]), step = self.T) 
					
				
				
		









		