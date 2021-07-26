from model.ELLA_non_stat import ELLA_non_stat
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression, Lasso
class NSELLA_GDCV(ELLA_non_stat):
	
	def get_ns_data(self,X=None,y=None):
		Xt = None
		yt = None
		if isinstance(X,np.ndarray):
			Xt = list(X.transpose(1,0,2))
		if isinstance(y,np.ndarray):
			yt = list(y.transpose())
		return Xt,yt

	def fit(self,X,y):
		# X, np array, X[i] = [task1 xi, task2 xi, ...]
		# y, np array, y[i] = [task1 yi, task2 yi, ...]
		Xt,yt = self.get_ns_data(X=X,y=y)
		T = len(Xt)
		for i in range(T):

			super().fit(Xt[i],yt[i],i)

	def single_task_predict(self, X, task_id):
		""" Output ELLA's predictions for the specified data on the specified
			task_id.  If using a continuous model (Ridge and LinearRegression)
			the result is the prediction.  If using a classification model
			(LogisticRgerssion) the output is currently a probability.
		"""
		if self.base_learner == LinearRegression or self.base_learner == Ridge:
			return (X@self.psig[task_id+1]@self.L@self.s_list[task_id])
		elif self.base_learner == LogisticRegression:

			return 1. / (1.0 + np.exp(-(X@self.psig[task_id+1]@self.L@self.s_list[task_id]))) > 0.5

	def predict(self,X):
		n = X.shape[0]
		T = X.shape[1]
		X,_ = self.get_ns_data(X=X) 
		yp = np.zeros([n,T])   

		for i in range(T):

			yp[:,i] = self.single_task_predict(X[i],i)[0]

		return yp

	def single_task_score(self, X, y, task_id):
			""" Output the score for ELLA's model on the specified testing data.
				If using a continuous model (Ridge and LinearRegression)
				the score is explained variance.  If using a classification model
				(LogisticRegression) the score is accuracy.
			"""

			return self.perf_metric(self.single_task_predict(X, task_id), y)
	def score(self,X,y):
		Xt,yt = self.get_ns_data(X=X,y=y)
		T = len(yt)
		acc = [self.single_task_score(Xt[i],yt[i],i) for i in range(T)]
		return np.mean(acc)

	def get_params(self,deep=False):
		return {'d': self.d,
				'k': self.k, 
				'base_learner': self.base_learner,
				'base_learner_kwargs': self.base_learner_kwargs, 
				'mu1': self.mu1,
				'mu3': self.mu3, 
				'lam': self.lam, 
				'max_iter': self.max_iter,
				'rhols': self.rhols,
				'window': self.window,
				'reg_weight': self.reg_weight,
				'writer': self.writer}

	def set_params(self, **params):
		if not params:
			# Simple optimization to gain speed (inspect is slow)
			return self
		valid_params = self.get_params(deep=True)

		for key, value in params.items():
			key, delim, sub_key = key.partition('__')
			if key not in valid_params:
				raise ValueError('Invalid parameter %s for estimator %s. '
								 'Check the list of available parameters '
								 'with `estimator.get_params().keys()`.' %
								 (key, self))

			if delim:
				nested_params[key][sub_key] = value
			else:
				setattr(self, key, value)
				valid_params[key] = value
		return self




