import cv2
import imutils
import struct
import numpy as np
from utils.pca import PCA_feature


def make_rotate_mnist_fig(T,angle,flatten=True,fixedn_train = None,fixedn_test = 1000,seed = None,angle0 = 0):

    if seed:
        np.random.seed(seed)

    with open('./mnist/train-labels.idx1-ubyte', 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        y_train = np.fromfile(flbl, dtype=np.int8)

    with open('./mnist/train-images.idx3-ubyte', 'rb') as fimg:
	    magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
	    X_train = np.fromfile(fimg, dtype=np.uint8).reshape(len(y_train), rows, cols)

    with open('./mnist/t10k-labels.idx1-ubyte', 'rb') as flbl:
	    magic, num = struct.unpack(">II", flbl.read(8))
	    y_test = np.fromfile(flbl, dtype=np.int8)

    with open('./mnist/t10k-images.idx3-ubyte', 'rb') as fimg:
	    magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
	    X_test = np.fromfile(fimg, dtype=np.uint8).reshape(len(y_test), rows, cols)

    Xs_train = []
    ys_train = []
    Xs_test = []
    ys_test = []

    for i in range(T):
        labels = np.array([np.random.randint(0, 5),np.random.randint(5, 10)])
        #print (labels)
        Xtr = X_train[np.isin(y_train, labels)]
        ytr = y_train[np.isin(y_train, labels)] == labels[0]
        Xte = X_test[np.isin(y_test, labels)]
        yte = y_test[np.isin(y_test, labels)] == labels[0]
        
        
        if fixedn_train:
            sample = np.array(np.random.choice(Xtr.shape[0], fixedn_train, replace=False))

            Xtr = Xtr[sample,:]
            ytr = ytr[sample]
            n_train = fixedn_train
        else:
            n_train = Xtr.shape[0]

        if fixedn_test:
            sample = np.array(np.random.choice(Xte.shape[0], fixedn_test, replace=False))
            Xte = Xte[sample,:]
            yte = yte[sample]
            n_test = fixedn_test
        else:
            n_test = Xte.shape[0]

        for j in range(n_train):
            Xtr[j] = imutils.rotate(Xtr[j],angle =angle0 + angle*i)  
            # if i == T-1:
            #     cv2.imshow("Image", Xtr[j])
            #     cv2.waitKey (0)
          
        for j in range(n_test):
            Xte[j] = imutils.rotate(Xte[j],angle=angle0 + angle*i)
        if flatten:
            Xtr = Xtr.reshape(n_train,-1)
            Xte = Xte.reshape(n_test,-1)
        Xs_train.append(Xtr)
        Xs_test.append(Xte)
        ys_train.append(ytr)
        ys_test.append(yte)

    return Xs_train, ys_train, Xs_test, ys_test

def make_meta_romnist_fig(T1,T2,stage1_train,stage2_train,angle,flatten=True,fixedn_test = 1000,seed = None):
    # Use tage1_train for T1 tasks and then stage2_train for the next T2 tasks
    # Designed for meta train using T1 tasks, and transfer knowledge for the next T2 tasks

    Xs_train1, ys_train1, Xs_test1, ys_test1 = make_rotate_mnist_fig(T1,
                                                                     angle,
                                                                     flatten=flatten,
                                                                     fixedn_train = stage1_train,
                                                                     fixedn_test = fixedn_test,
                                                                     seed = seed)
    Xs_train2, ys_train2, Xs_test2, ys_test2 = make_rotate_mnist_fig(T2,
                                                                     angle,
                                                                     flatten=flatten,
                                                                     fixedn_train = stage2_train,
                                                                     fixedn_test = fixedn_test,
                                                                     seed = seed+1,
                                                                     angle0 = T1 * angle)
    Xs_train = Xs_train1 + Xs_train2
    ys_train = ys_train1+(ys_train2)
    Xs_test = Xs_test1+(Xs_test2)
    ys_test = ys_test1+(ys_test2)
    #print (Xs_train[-1].shape)
    return Xs_train, ys_train, Xs_test, ys_test






def prepare_mnist(Tl,angle,n_trainl,seed,feature_type,method='classical'):
    # T, n_train are list for meta train
    if method =='classical':
        T = Tl
        Xs_train,ys_train,Xs_test,ys_test = make_rotate_mnist_fig(T,angle,fixedn_train=n_trainl,seed=seed)
    if method == 'meta':
        Xs_train,ys_train,Xs_test,ys_test = make_meta_romnist_fig(Tl[0],Tl[1],n_trainl[0],n_trainl[1],angle,seed=seed)
        T = sum(Tl)


    Xs_train_pca, Xs_test_pca = PCA_feature(Xs_train,Xs_test,feature_type,seed = seed)

    #print ('PCA feature done. Feature type: ' + feature_type )
    Xs_train_pca = [np.hstack([Xs_train_pca[i], np.ones((Xs_train_pca[i].shape[0],1))]) for i in range(T)]
    Xs_test_pca = [np.hstack([Xs_test_pca[i], np.ones((Xs_test_pca[i].shape[0],1))]) for i in range(T)]
    d = Xs_train_pca[0].shape[1]

    #print ('Feature Dimension = ', d)

    return Xs_train_pca, ys_train, Xs_test_pca, ys_test, d


