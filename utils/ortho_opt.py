import numpy as np
from numpy.linalg import *
from scipy.linalg import qr,norm

import tensorflow as tf

def ortho_opt(psi,h,g,method,writer = None):
	# A Feasible Method for Optimization with Orthogonality Constraints
	# Update scheme for BB step size ['QR', 'curvilinear', 'curvilinear_rank1']
	# Recommand max_iter 50,50,500,

	# EFFICIENT RIEMANNIAN OPTIMIZATION ON THE STIEFEL MANIFOLD VIA THE CAYLEY TRANSFORM
	# Update scheme with an iterative estimation of the Cayley transform [CayleySGD, CayleyADAM]
	# Under construction


	def QR_update(XX):
		# orthogonal update by qr	
		X = np.matrix(XX)
		Q, RR = qr(X)
		X = Q@np.sign(RR * np.identity(d))

		return X

	def CV_rank1_update(X,tau):
		# Wen13 Lemma5
		X = np.matrix(X)
		G = g(X)

		val = [np.trace(X@G.T@(G[:,i]@X[:,i].T - X[:,i]@G[:,i].T)) for i in range(X.shape[1])]
		q = val.index(max(val))
		W = G[:,q] @ X[:,q].T - X[:,q] @ G[:,q].T
		ytau = rank1inv(G[:,q],X[:,q],tau/2)@(np.identity(X.shape[0]) - tau/2*W) @ X
		return ytau

	def Cayley_update(X,tau):
		# Jun20
		X = np.matrix(X)
		G = g(X)
		W = G@X.T - X@G.T
		ytau = X + tau/2 * W @(X + ytau)
		return ytau



	def check_ortho(XX):
		tiny = 1e-13
		X = np.matrix(XX)
		if norm(X.T@X - np.identity(d)) > tiny:
			X = QR_update(X)
		return X

	def iprod(x,y):
		a = (x.T*y).sum()
		return a

	def rank1inv(a,b,alpha):
		# a,b, vector, inv(I + alpha(a@b.T-b@a.T))
		d = a.shape[0]
		a = np.matrix(a)
		b = np.matrix(b)
		A = (b.T@a)[0,0]
		B = (b.T@b)[0,0]
		C = (a.T@a)[0,0]
		x1 = -alpha * (1-alpha * A)/(1-alpha**2*A**2 + alpha**2*B*C)
		x2 = alpha * (1 + alpha * A)/(1-alpha**2*A**2 +alpha**2*B*C)
		x3 = -alpha**2*B/(1-alpha**2*A**2 +alpha**2*B*C)
		x4 = -alpha**2*C/(1-alpha**2*A**2 + alpha**2*B*C)
		invm = np.identity(d) + x1* a@b.T + x2 * b@a.T + x3*a@a.T + x4*b@b.T
		return invm

	def BBstepscale(X,tau,Q,Cval,method):
		nls = 1
		rhols  = 1e-4
		eta = 0.1

		X = np.matrix(X)
		G = g(X)	
		GX = G.T@X
		dtX = G - X@GX
		GXT = G@X.T
		H = 0.5*(GXT - GXT.T)
		RX = H@X
		nrmG  = norm(dtX, 'fro')
		deriv = rhols*nrmG**2; #deriv
		while 1:
			# calculate G, F,        
			if method == 'curvilinear':
				Y = solve(np.identity(d) + tau*H, X - tau*RX)
			elif method == 'QR':
				Y = QR_update(X - tau*dtX)
			elif method =='curvilinear_rank1':
				Y = CV_rank1_update(X,tau)

			
			Y = check_ortho(Y)
			F = h(Y)
			G = g(Y)
		   
			if F <= Cval - tau*deriv or nls >= 50:
				return Y,tau

			tau = eta*tau
			nls = nls+1

	def writelog(writer,X,itr):
		G = g(X)
		GX = G.T@X
		dtX = G - X@GX
		nrmG  = norm(dtX, 'fro')

		with writer.as_default():  
			tf.summary.scalar('psi_loss',h(X), step = itr)
			tf.summary.scalar('psi_gradient',nrmG, step = itr)




	def BB_step(X,method,max_iter):

		xtol = 1e-6
		gtol = 1e-10
		ftol = 1e-12


		# parameters for control the linear approximation in line search,
		tau  = 1e2
		gamma  = 0.85
		nt  = 3
		mxitr  = max_iter
		crit = np.zeros((nt,3))

		## Initial function value and gradient
		# prepare for iterations
		F = h(X)
		G = g(X)
		GX = G.T@X
		dtX = G - X@GX
		nrmG  = norm(dtX, 'fro')
		  
		Q = 1
		Cval = F


		## main iteration
		for itr in range(1,mxitr):
			XP = X
			FP = F
			GP = G
			dtXP = dtX

			# scale step size
			X,tau = BBstepscale(X,tau,Q,Cval,method)

			F = h(X)
			G = g(X)
			GX = G.T@X
			dtX = G - X@GX
			nrmG  = norm(dtX, 'fro')   

			print (itr, F, nrmG, tau)


			S = X - XP
			XDiff = norm(S,'fro')/np.sqrt(d)
			FDiff = abs(FP-F)/(abs(FP)+1)
			

			Y = dtX - dtXP
			SY = abs(iprod(S,Y))
			if itr%2==0:
				tau = (norm(S,'fro')**2)/SY
			else:
				tau  = SY/(norm(Y,'fro')**2)
			tau = max(min(tau, 1e15), 1e-15)
			

			
			crit[itr%3,:] = [nrmG, XDiff, FDiff]
			mcrit = np.mean(crit,0)
			if ( XDiff < xtol and FDiff < ftol ) or nrmG < gtol or all(mcrit[2:3] < 10*[xtol, ftol]):
				break
		 
			Qp = Q
			Q = gamma*Qp + 1
			Cval = (gamma*Qp*Cval + F)/Q

			if writer:
				writelog(writer,X,itr)

		X = check_ortho(X)
		return X


	def step(X,method,max_iter):

		xtol = 1e-6
		gtol = 1e-10
		ftol = 1e-12


		# parameters for control the linear approximation in line search,
		tau  = 1e2
		gamma  = 0.85
		nt  = 3
		mxitr  = max_iter
		crit = np.zeros((nt,3))

		## Initial function value and gradient
		# prepare for iterations
		F = h(X)
		G = g(X)
		GX = G.T@X
		dtX = G - X@GX
		nrmG  = norm(dtX, 'fro')
		  
		Q = 1
		Cval = F


		## main iteration
		for itr in range(1,mxitr):
			XP = X
			FP = F
			GP = G
			dtXP = dtX

			# scale step size
			X,tau = BBstepscale(X,tau,Q,Cval,method)

			F = h(X)
			G = g(X)
			GX = G.T@X
			dtX = G - X@GX
			nrmG  = norm(dtX, 'fro')   

			print (itr, F, nrmG, tau)


			S = X - XP
			XDiff = norm(S,'fro')/np.sqrt(d)
			FDiff = abs(FP-F)/(abs(FP)+1)
			

			Y = dtX - dtXP
			SY = abs(iprod(S,Y))
			if itr%2==0:
				tau = (norm(S,'fro')**2)/SY
			else:
				tau  = SY/(norm(Y,'fro')**2)
			tau = max(min(tau, 1e15), 1e-15)
			

			
			crit[itr%3,:] = [nrmG, XDiff, FDiff]
			mcrit = np.mean(crit,0)
			if ( XDiff < xtol and FDiff < ftol ) or nrmG < gtol or all(mcrit[2:3] < 10*[xtol, ftol]):
				break
		 
			Qp = Q
			Q = gamma*Qp + 1
			Cval = (gamma*Qp*Cval + F)/Q

			if writer:
				writelog(writer,X,itr)

		X = check_ortho(X)
		return X


	# def CayleyAdam(X,max_iter):
	# 	beta1 = 0.6
	# 	beta2 = 0.2
	# 	ep = 1e-8 
	# 	q = 5e-1
	# 	s = 2
	# 	M = 0 * X
	# 	v = 1
	# 	l = 1e-1
	# 	for i in range(1,max_iter):
	# 		G = self.g(X)
	# 		M = beta1 * M + (1 - beta1) * G
	# 		v = beta2 * v + (1 - beta2) * norm(G)**2
	# 		v1 = v /(1 - beta2 ** i )
	# 		r = (1 - beta1 ** i )* np.sqrt(v1 + ep)
	# 		W_hat = M@X.T - 1/2*X@(X.T@M@X.T)
	# 		W = (W_hat - W_hat.T)/r
	# 		M = r*W@X
	# 		a = min(l,2*q/(norm(W) + ep))
	# 		#print (self.T, a,norm(M),norm(X))
	# 		Y = X +  a*M
	# 		for j in range(s):
	# 			Y = X + a/2* W @( X  + Y  )
	# 		if abs(self.h(X) - self.h(Y)) < 1e-14:
	# 			X = Y 
	# 			break
	# 		X = Y
	# 		#print (self.T, i,norm(G - X@G.T@X),self.h(X))

	# 	return X

	# def CayleySGD(X):
	# 	beta = 0.1
	# 	ep = 1e-8 
	# 	q = 5e-1
	# 	s = 2
	# 	l = 1e-2
	# 	M = 0 * X
	# 	for i in range(1,max_iter):
	# 		G = self.g(X)
	# 		M = beta*M - G
	# 		W_hat =  M@X.T - 1/2*X@(X.T@M@X.T)
	# 		W = (W_hat - W_hat.T)
	# 		a = min(l,2*q/(norm(W) + ep))
	# 		Y = X +  a*M
	# 		for j in range(s):
	# 			Y = X + a/2* W @( X  + Y  )
	# 		if abs(self.h(X) - self.h(Y)) < 1e-14:
	# 			X = Y 
	# 			break
	# 		X = Y
	# 		G = self.g(X)

	# 		if self.writer:
	# 			writelog(writer,X,i)


	# 		#print (self.T, i,norm(G - X@G.T@X),self.h(X))

	# 	return X

	d = psi.shape[0]
	max_iter = 40
	return BB_step(psi,method,max_iter)





		
		