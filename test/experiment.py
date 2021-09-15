import numpy as np
import sys
import tensorflow as tf 
import datetime
sys.path.append("./")
from problem.classification_problems import classical_syn_cls,classical_mnist,meta_syn_cls,meta_mnist
import matplotlib.pyplot as plt

# Choose which problem to run
problem = 'classification'
data_style = 'classical'
T = 20
n_train = 100


# Choose the classifier
logistic = True
ella = True
non_stat = True

# Choose the output way
write = True
plot = False
print_res = False

# iteration parameter
total_iter = 1
count = 0
cur = 0

# Model parameter for all

seed = 1
feature_type = 'separate'
# Model parameter for ella and non-ella
regs = 0.3
writer = None
window = None
if data_style == 'meta':
    T = [20,10]
    n_train = [100,20]

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
date =  datetime.date.today().strftime("%y%m%d")
total_ella = []
total_non = []
total_log = []
x_axis = []



kwargs = {'n_train'     : n_train, 
          'window'      : window, 
          'logistic'    : logistic, 
          'ella'        : ella, 
          'non_stat'    : non_stat,
          'seed'        : seed,
          'feature_type': feature_type,
          'T'           : T,
          'reg_weight'  : regs}



if problem == 'classification':
    kwargs['d'] = 10
    kwargs['k'] = 5
    if data_style == 'classical':
        model = classical_syn_cls
    if data_style == 'meta':
        model = meta_syn_cls
if problem == 'mnist':
    kwargs['angle'] = 5   
    if data_style == 'meta':
        model = meta_mnist
    if data_style == 'classical':
        model = classical_mnist



np.random.seed(seed)








while count < total_iter:
    cur += 1
    # Do the iteration for different variables by changing x = cur
    

    # set Tensroboard
    log_dir = f'results/logs_{problem}/{date}/w{window}n{n_train}/{current_time}'
    if write:
        writer = tf.summary.create_file_writer(log_dir)
    kwargs['writer'] = writer


    #try:
        #n_train = cur*10
        #kwargs['n_train'] = n_train
    print(f'{problem}, training size = {n_train}, window = {window}')    
    run_mdoel = model(**kwargs)    
    acc_log, acc_ella, acc_non_stat = run_mdoel.run()   
    if ella:
        total_ella.append(acc_ella[-1]) 
    if non_stat:
        total_non.append(acc_non_stat[-1]) 
    if logistic:
        total_log.append(acc_log[-1])
    count += 1
    if plot:
        x_axis = [i for i in range(sum(T))]
        x_label = 'task'
    # except ValueError:
    #     print ('value error')
    #     continue


if write and non_stat:    
    writer = tf.summary.create_file_writer(log_dir)
    with writer.as_default():  
        tf.summary.scalar('acc_non_stat_avg',np.mean(total_non), step = cur)
        tf.summary.scalar('std_non_stat',np.std(total_non, ddof=1), step = cur)

if print_res:        
    ella and print(f"ELLA: {np.mean(total_ella)*100:.2f} \% \pm {np.std(total_ella, ddof=1)*100 :.2f}\%")
    non_stat and print(f"Non: {np.mean(total_non)*100:.2f} \% \pm {np.std(total_non, ddof=1)*100 :.2f}\%")    
    logistic and print(f"Log: {np.mean(total_log)*100:.2f} \% \pm {np.std(total_log, ddof=1)*100 :.2f}\%")


if plot:
    plt.figure()
    ella and plt.plot(x_axis,total_ella,label='ELLA')
    non_stat and plt.plot(x_axis,acc_non_stat,label='ELLA_NS')
    logistic and plt.plot(x_axis, acc_log,label='Logistic')
    plt.xlabel(x_label)
    plt.ylabel('Average Accuracy')
    plt.legend()
    plt.title(f'Task {T}, Training size {n_trainl}')
    plt.savefig(f'../n{n_trainl[0]}T{T[0]}.png')
