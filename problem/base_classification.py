from sklearn.linear_model import LogisticRegression
import numpy as np
import sys
import random
import tensorflow as tf
sys.path.append("./")
from model.ELLA_non_stat import ELLA_non_stat
from model.ELLA import ELLA
from utils.make_rotate_mnist_data import prepare_mnist
from scipy.linalg import sqrtm, inv, norm, qr

import warnings
warnings.filterwarnings("ignore")



class base_classification(object):
	def __init__(self,
				 seed=1,
				 T = 20,
				 angle = 5,
				 k = 3,
				 d = 10,
				 n_train = 500,
				 max_iter= 500,
				 window = None,
				 feature_type = 'shared',
				 reg_weight = 1,
				 logistic = True,
				 ella = True,
				 non_stat = True,
				 writer = None):

		self.seed = seed
		self.T = T
		self.angle = angle
		self.k = k 
		self.d = d
		self.n_train = n_train
		self.max_iter = max_iter
		self.window = window
		self.feature_type = feature_type
		self.reg_weight = reg_weight
		self.logistic = logistic
		self.ella = ella 
		self.non_stat = non_stat
		self.writer = writer
		random.seed(self.seed)
		np.random.seed(self.seed)
		self.base_learner_kwargs =  {'max_iter':10000,  'solver':'liblinear', 'C': 1}
		self.a = None
		self.L = None
		self.s = None

	def measure(self,acc):
		return [sum(acc[:i+1])/(i+1) for i in range(self.T)]

	

	def run(self):
		acc_log, acc_ella, acc_non_stat = [],[],[]
		self.prepare_data()
		if isinstance(self.T, list):
			T_count = sum(self.T)
		else:
			T_count = self.T
		if self.logistic:
			print ('==Logistic Regression==')
			acc = []
			model_logistic = LogisticRegression(fit_intercept = False, **self.base_learner_kwargs)
			for t in range(T_count):
				model_logistic.fit(self.Xs_train[t], self.ys_train[t])
				acc.append(model_logistic.score(self.Xs_test[t], self.ys_test[t]))
			acc_log = self.measure(acc)
			#print (acc)
			print (f'Seed:{self.seed}. Average accuracy: {acc_log[-1]}')

		if self.ella:
			print ('==ELLA==')
			acc = []
			model_ella = ELLA(self.d,
							  self.k,
							  LogisticRegression,
							  self.base_learner_kwargs,
							  mu = 0.001,
							  lam = 1e-3,
							  window = self.window,
							  writer = self.writer)
			for t in range(T_count):
				model_ella.fit(self.Xs_train[t], self.ys_train[t], t)
			for t in range(T_count):
				acc.append(model_ella.score(self.Xs_test[t], self.ys_test[t], t))
			acc_ella= self.measure(acc)
			print (f'Seed:{self.seed}. Average accuracy: {acc_ella[-1]}')

		if self.non_stat:
			print ('==non-stat-ELLA==')
			acc = []
			model_non_ella = ELLA_non_stat(self.d,
										   self.k,
										   LogisticRegression,
										   self.base_learner_kwargs,
										   window=self.window,
										   max_iter=self.max_iter,
										   reg_weight=self.reg_weight,
										   writer = self.writer,
										   seed = self.seed,
										   true_psi = self.a,
										   true_L = self.L,
										   true_s = self.s)

			for t in range(T_count):
				model_non_ella.fit(self.Xs_train[t], self.ys_train[t], t)
				if t > 5:
					print (f'**task {t} train_acc: ', model_non_ella.score(self.Xs_train[t], self.ys_train[t], t))
			for t in range(T_count):
				acc.append(model_non_ella.score(self.Xs_test[t], self.ys_test[t], t))

			acc_non_stat = self.measure(acc)
			for t in range(T_count):
				if self.writer:
					with self.writer.as_default():  
						tf.summary.scalar('diff_ture_psi_fnorm',norm(model_non_ella.psi_list[t] - self.a),step = t)
						tf.summary.scalar('avg_acc',acc_non_stat[t], step = t)
						tf.summary.scalar('test_acc',acc[t], step = t)
						tf.summary.scalar('train_acc',model_non_ella.score(self.Xs_train[t], self.ys_train[t], t), step = t)
			# print (acc)
			print (f'Seed:{self.seed}. Average accuracy: {acc_non_stat[-1]}')

		return acc_log, acc_ella, acc_non_stat





		



