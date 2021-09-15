from args import parse_args
import numpy as np


def run(args):
	if args.writer:
	    import tensorflow as tf 
	    import datetime
	    date =  datetime.date.today().strftime("%y%m%d")
	    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	    log_dir = f'results/logs_{args.dataset}/{date}/{current_time}'
	    writer = tf.summary.create_file_writer(log_dir)
	else:
	    writer = None
	    
	    
	    



	# model
	if args.base_learner == 'LR':
	    from sklearn.linear_model import LogisticRegression
	    baselearner = LogisticRegression
	elif args.base_learner == 'Ridge':
	    import sklearn.linear_model.Ridge as baselearner
	else: 
	    raise NotImplementedError(f'Unknown base learner {args.base_learner}')
	    
	    
	base_learner_kwargs =  {'max_iter':10000,  'solver':'liblinear', 'C': 1, 'random_state': args.seed}
	if 'NS' in args.baseline:
	    from model.ELLA_non_stat import ELLA_non_stat
	    model = ELLA_non_stat(args,baselearner, base_learner_kwargs,writer,true_p,true_L)
	elif args.baseline == 'ELLA':
	    from model.ELLA import ELLA
	    model = ELLA(args.d,args.k,baselearner,base_learner_kwargs,mu = args.mu1,lam = args.mu3)
	elif args.baseline == 'LR':
	    from sklearn.linear_model import LogisticRegression
	    model = LogisticRegression(fit_intercept = False, **base_learner_kwargs)
	else:
	    raise NotImplementedError(f'Unknown baseline {args.baseline}')


	acc = []

	if args.data_style == 'classical':
	    T = args.T
	else:
	    T = args.T + args.T_meta

	for t in range(T):
	    if args.baseline == 'LR':
	        model.fit(Xs_train[t], ys_train[t])
	        acc.append(model.score(Xs_test[t], ys_test[t]))
	    else:
	        model.fit(Xs_train[t], ys_train[t], t)
	        acc.append(np.mean([model.score(Xs_test[i], ys_test[i],i) for i in range(t+1)]))

	return acc[-1]


args = parse_args()
np.random.seed(args.seed)
# data
if args.dataset == 'synthetic':
    from utils.data_process import prepare_syncls
    Xs_train,ys_train,Xs_test,ys_test,true_p,true_L = prepare_syncls(args)
elif args.dataset == 'mnist':
    from utils.make_rotate_mnist_data import prepare_mnist
    Xs_train,ys_train,Xs_test,ys_test, args.d = prepare_mnist(args)
    true_p = None
    true_L = None
else: 
    raise NotImplementedError(f'Unknown dataset {args.dataset}')


grid_epoch = 1
best_acc = 0
mu1 = [1e-3, 1e-4, 1e-5,  1e-6, 1e-7, 1e-8]
mu3 = [1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5,  1e-6]
mu2 = [1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5,  1e-6]
reg_weight = [1e-3, 3e-3, 6e-3, 1e-2,  6e-2, 1e-1, 3e-1, 6e-1,1,'avg']
for i in range(grid_epoch):
	args.mu1 = np.random.choice(mu1)
	args.mu2 = np.random.choice(mu2)
	args.mu3 = np.random.choice(mu3)
	reg = np.random.choice(reg_weight)
	print ([args.mu1,args.mu2,args.mu3,reg])
	if reg == 'avg':
		args.reg_method = 'avg'
	else:
		args.reg_method = 'ewma'
		args.reg_weight = float(reg)

	acc = run(args)

	if acc > best_acc:
		best_para = [args.mu1,args.mu2,args.mu3,reg]
		best_acc = acc
print (best_acc, best_para)

	        
