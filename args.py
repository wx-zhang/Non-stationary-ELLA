import argparse
import yaml

def parse_args():
	help_formatter = argparse.ArgumentDefaultsHelpFormatter
	parser = argparse.ArgumentParser('NS_ELLA', formatter_class=help_formatter)

	group = parser.add_argument_group('General setting')
	group.add_argument('--seed', type=int, default=0, help='Random seed for numpy and scikit-learn')
	baselines = ['NS_ELLA_accall', 'NS_ELLA_accL', 'NS_ELLA_accp', 'NS_ELLA_accpavg', 'NS_ELLA','ELLA','LR']
	group.add_argument('--baseline', default='NS_ELLA',choices=baselines)
	group.add_argument('--dataset', choices=['synthetic','mnist'], default='synthetic')
	group.add_argument('--data_style', choices=['classical', 'meta'], default='classical')
	group.add_argument('--n_train_classical',type=int,default=100,help='Training size in the classical leraning stage')
	group.add_argument('--n_train_meta', type=int, default=20,help='Training size in the transfer stage')

	group = parser.add_argument_group("Hyper-parameters")
	group.add_argument('--d', type = int, default=10, help='Input dimension')
	group.add_argument('--k', type = int, default=5, help='Number of components of latent space')
	group.add_argument('--mu1', type=float, default=1e-3,help='Regularization coefficient of s')
	group.add_argument('--mu2', type=float,default=1e-3,help='Regularization coefficient of transition matrix')
	group.add_argument('--mu3', type=float, default=1e-3,help='Regularization coefficient of latent space')
	group.add_argument('--reg_method',choices=['avg','ewma'],default='ewma')
	group.add_argument('--reg_weight',default=1e-3, type=float, help='Coefficient of the average transition matrix in exponentially weighted moving average method')

	group = parser.add_argument_group("Initilization")
	group.add_argument('--k_init', type=bool, default=True, help='Use first k tasks to init latent space')
	group.add_argument('--multi_init',type=bool, default=True,help='Multi-taks init for the regularization term of transition matrix')
	group.add_argument('--n_multi_init', type=int, default=5,help='Use first n tasks to multi task init transition matrix')
	group.add_argument('--init_epoch', type=int, default=50, help='Epochs for the multi task init')

	group = parser.add_argument_group('General learning setting')
	group.add_argument('--T', type=int,default=20,help='Total task number')
	group.add_argument('--T_meta',type=int,default=20,help='extra tasks after meta training')
	group.add_argument('--true_p',type=bool, default=False, help='Use true transition in the learning')
	group.add_argument('--true_pavg',type=bool, default=False, help='Use true average transition in the learning')
	group.add_argument('--true_L',type=bool, default=False, help='Use true latent space in the learning')
	group.add_argument('--stop_update_L',type=int, default=1000, help='Stop updating latent space after some tasks')
	group.add_argument('--stop_update_p',type=int, default=1000, help='Stop updating transition after some tasks')
	group.add_argument('--base_learner', type=str, default= 'LR', choices=['LR','Ridge'], help='Base learning algorithm for the optimal parameter alpha')

	group = parser.add_argument_group('Orthogonal optimization setting')
	group.add_argument('-w', '--window', type=int, default=None, help='Use only the loss of last w tasks for optimizations')
	group.add_argument('--max_iter', type=int,default=50,help='Maximum iteration of intervaive method except pymanopt')
	group.add_argument('--rhols', type=float, default=1e-5, help='Scale of derivative in BB step')
	group.add_argument('--opt_method', default='pymanopt', choices=['pymanopt', 'QR', 'curvilinear', 'curvilinear_rank1'])

	group = parser.add_argument_group('Log')
	group.add_argument('--writer', type=bool, default=False, help='Write log use tensorboard')
	group.add_argument('--save', type=bool, default=False,help='save average accuracy array')

	args = parser.parse_args()

	args.model_config = f"config/{args.baseline}.yaml"
	try:
		with open(args.model_config) as f:
			file_args = yaml.unsafe_load(f)
			# overwrite the default values with the values from the file.
			args_dict = vars(args)
			args_dict.update(vars(file_args))
			args = argparse.Namespace(**args_dict)
	except FileNotFoundError:
		pass

	if args.dataset == 'synthetic':
		args.T = 100
		args.k = 5
		args.d = 10
	elif args.dataset == 'mnist':
		args.T = 20
		args.k = 3
		args.n_train_classical = 50


	return args



