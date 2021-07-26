from problem.base_classification import base_classification
from utils.data_process import prepare_syncls
from utils.make_rotate_mnist_data import prepare_mnist
import numpy as np



class classical_syn_cls(base_classification):

	def prepare_data(self):
		self.Xs_train,self.ys_train,self.Xs_test,self.ys_test,self.a,self.L,self.s = prepare_syncls(self.seed,self.k,self.d,self.T,self.n_train)


class classical_mnist(base_classification):

	def prepare_data(self):
		self.Xs_train,self.ys_train,self.Xs_test,self.ys_test, self.d = prepare_mnist(self.T,self.angle,self.n_train,self.seed,self.feature_type)

class meta_classification(base_classification):


	def measure(self, acc):
		acc1 = [np.mean(acc[:t+1]) for t in range(self.T[0])]
		acc2 = [np.mean(acc[self.T[0]:self.T[0]+t+1]) for t in range(self.T[1])]

		return acc1 + acc2

class meta_syn_cls(meta_classification):
	def prepare_data(self):

		self.Xs_train,self.ys_train,self.Xs_test,self.ys_test,self.a,self.L,self.s = prepare_syncls(self.seed,self.k,self.d,self.T,self.n_train,method='meta')


class meta_mnist(meta_classification):


	def prepare_data(self):
		self.Xs_train,self.ys_train,self.Xs_test,self.ys_test,self.d = prepare_mnist(self.T,self.angle,self.n_train,self.seed,self.feature_type,method='meta')
