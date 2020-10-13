# Author: Guiming Zhang
# Last update: Oct 13 2020

import os, time, sys
import numpy as np
from sklearn.model_selection import train_test_split
import json
import matplotlib
#matplotlib.use('Qt4Agg')
from matplotlib import pyplot as plt
from pathos.multiprocessing import ProcessingPool as Pool

from IPython import display
from sklearn.metrics import roc_auc_score
import mxnet as mx
from mxnet import nd, gluon, init, autograd
from mxnet.gluon import nn
from mxnet import context
import sparse ## https://github.com/mrocklin/sparse
#sys.path.insert(0, 'utilities')
import gdalwrapper, covariates_config

def read_data_modeling(occurrences_fns, background_fns, preprocessed=False, cov_stats_fn=None, cov_config_fn=None):
    '''Read in data from data files
       Data points are at the center of neighborhood (5 x 5 by default)
       Data files are in .npz format (numpy compressed format)
       preprocessed: have feature data been standardized and nansfilled?
    '''
    X_occur, y_occur = None, None
    X_back, y_back = None, None

    index_selected_var = None
    if cov_config_fn is not None:
        config = covariates_config.covariates_config(cov_config_fn)
        index_selected_var = config.get_selected_index()

    for occurrences_fn in occurrences_fns:
        ## load features at occurrence locations from compressed file
        with np.load(occurrences_fn) as fdata:
            tmp = fdata[fdata.files[0]]
            tmp = tmp[:,:,int(tmp.shape[2]/2),int(tmp.shape[3]/2)]

            if index_selected_var is not None:
                tmp = tmp[:,index_selected_var]

            if X_occur is None:
                X_occur = tmp
                #print('X_occur shape:', X_occur.shape)
                y_occur = np.ones((tmp.shape[0],1))
                #print('y_occur shape:', y_occur.shape)
            else:
                X_occur = np.concatenate((X_occur, tmp),axis=0)
                y_occur = np.concatenate((y_occur, np.ones((tmp.shape[0],1))),axis=0)

    for background_fn in background_fns:
        ## load features at background locations from compressed file
        with np.load(background_fn) as fdata:
            tmp = fdata[fdata.files[0]]
            tmp = tmp[:,:,int(tmp.shape[2]/2),int(tmp.shape[3]/2)]

            if index_selected_var is not None:
                tmp = tmp[:,index_selected_var]

            if X_back is None:
                X_back = tmp
                #print('X_back shape:', X_back.shape)
                y_back = np.zeros((X_back.shape[0],1))
                #print('y_back shape:', y_back.shape)
            else:
                X_back = np.concatenate((X_back, tmp),axis=0)
                y_back = np.concatenate((y_back, np.zeros((tmp.shape[0],1))),axis=0)

    ## combine occurrence locations and background locations
    occur = np.concatenate((X_occur, y_occur), axis=1)
    back = np.concatenate((X_back, y_back), axis=1)
    data = np.concatenate((occur, back), axis = 0)

    if not preprocessed and cov_stats_fn is not None:
        ## standardize with mean and std per covariate
        stats = np.loadtxt(cov_stats_fn, delimiter=',')

        if index_selected_var is not None:
            stats = stats[index_selected_var,:]

        means = stats[:,2]
        stds = stats[:,3]
        data[:,0:-1] = (data[:,0:-1] - means) / stds

    if not preprocessed:
        ## fill nans
        col_mean = np.nanmean(data[:,0:-1], axis=0)
        #Find indices that you need to replace
        inds = np.where(np.isnan(data[:,0:-1]))
        #Place column means in the indices. Align the arrays using take
        data[:,0:-1][inds] = np.take(col_mean, inds[1])

    ## shuffle records
    np.random.shuffle(data)

    return data

def read_data_modeling_cnn(occurrences_fns, background_fns, nbrhd_size=5, preprocessed=False, cov_stats_fn=None, cov_config_fn = None):
    '''Read in data from data files
       Data points are at the centers of neighborhoods or in the whole neighborhood (5 x 5 by default)
    '''
    try:
        X_occur, y_occur, coords_occur = None, None, None
        X_back, y_back, coords_back = None, None, None

        index_selected_var = None
        if cov_config_fn is not None:
            config = covariates_config.covariates_config(cov_config_fn)
            index_selected_var = config.get_selected_index()

        for occurrences_fn in occurrences_fns:
            ## load features at occurrence locations from compressed file
            with np.load(occurrences_fn) as fdata:
                tmp = fdata[fdata.files[0]]
                xys = fdata[fdata.files[1]]

                if tmp.shape[2] < nbrhd_size or tmp.shape[3] < nbrhd_size:
                    print('Neighborhood size is larger than offered in data files. Exiting...')
                    sys.exit(1)
                else:
                    center_x = int(tmp.shape[2]/2)
                    center_y = int(tmp.shape[3]/2)
                    wd = int(nbrhd_size/2)
                    tmp = tmp[:, :, center_x-wd:center_x+wd+1, center_y-wd:center_y+wd+1]

                if index_selected_var is not None:
                    tmp = tmp[:,index_selected_var,:,:]

                if X_occur is None:
                    X_occur = tmp
                    #print('X_occur shape:', X_occur.shape)
                    coords_occur = xys
                    y_occur = np.ones((tmp.shape[0], 1, tmp.shape[2], tmp.shape[3]))
                    #print('y_occur shape:', y_occur.shape)
                else:
                    X_occur = np.concatenate((X_occur, tmp),axis=0)
                    coords_occur = np.concatenate((coords_occur, xys),axis=0)
                    y_occur = np.concatenate((y_occur, np.ones((tmp.shape[0], 1, tmp.shape[2], tmp.shape[3]))),axis=0)

        for background_fn in background_fns:
            ## load features at background locations from compressed file
            with np.load(background_fn) as fdata:
                tmp = fdata[fdata.files[0]]
                xys = fdata[fdata.files[1]]

                if tmp.shape[2] < nbrhd_size or tmp.shape[3] < nbrhd_size:
                    print('Neighborhood size is larger than offered in data files. Exiting...')
                    sys.exit(1)
                else:
                    center_x = int(tmp.shape[2]/2)
                    center_y = int(tmp.shape[3]/2)
                    wd = int(nbrhd_size/2)
                    tmp = tmp[:, :, center_x-wd:center_x+wd+1, center_y-wd:center_y+wd+1]

                if index_selected_var is not None:
                    tmp = tmp[:,index_selected_var,:,:]

                if X_back is None:
                    X_back = tmp
                    #print('X_back shape:', X_back.shape)
                    coords_back = xys
                    y_back = np.zeros((tmp.shape[0], 1, tmp.shape[2], tmp.shape[3]))
                    #print('y_back shape:', y_back.shape)
                else:
                    X_back = np.concatenate((X_back, tmp),axis=0)
                    coords_back = np.concatenate((coords_back, xys),axis=0)
                    y_back = np.concatenate((y_back, np.zeros((tmp.shape[0], 1, tmp.shape[2], tmp.shape[3]))),axis=0)

        ## combine occurrence locations and background locations
        #occur = np.concatenate((X_occur, y_occur), axis=1)
        #back = np.concatenate((X_back, y_back), axis=1)
        #data = np.concatenate((occur, back), axis = 0)
        coords = np.concatenate((coords_occur, coords_back), axis = 0)
        X = np.concatenate((X_occur, X_back), axis = 0)
        y = np.concatenate((y_occur, y_back), axis = 0)

        ## standardize with mean and std per covariate
        if not preprocessed and cov_stats_fn is not None:
            stats = np.loadtxt(cov_stats_fn, delimiter=',')

            if index_selected_var is not None:
                stats = stats[index_selected_var,:]

            means = []
            stds = []
            for i in range(X_occur.shape[1]):
                means.append(np.ones((X.shape[2],X.shape[3]))*stats[i,2])
                stds.append(np.ones((X.shape[2],X.shape[3]))*stats[i,3])
            means = np.array(means)
            stds = np.array(stds)
            X = (X - means)/stds

        if not preprocessed:
            ## fill nans
            col_mean = np.nanmean(X, axis=0)
            #Find indices that you need to replace
            inds = np.where(np.isnan(X))
            #Place column means in the indices. Align the arrays using take
            X[inds] = np.take(col_mean, inds[1])

        ## combine X, y
        Xy = np.concatenate((X, y), axis=1)

        ## shuffle records
        idx = np.array(range(Xy.shape[0]))
        np.random.shuffle(idx)

        coords = coords[idx]
        Xy = Xy[idx]

        return coords, Xy

    except (Exception) as error:
        print(error)
        raise error

def read_data_prediction(prediction_pnts_fns, preprocessed=False, cov_stats_fn=None, cov_config_fn=None):
    '''Read in features at prediction points from data files
       Data points are at the center of neighborhood (5 x 5 by default)
    '''
    X_pred = None
    coords_pred = None

    index_selected_var = None
    if cov_config_fn is not None:
        config = covariates_config.covariates_config(cov_config_fn)
        index_selected_var = config.get_selected_index()

    for prediction_pnts_fn in prediction_pnts_fns:
        ## load features at prediction locations from compressed file
        with np.load(prediction_pnts_fn) as fdata:
            tmp = fdata[fdata.files[0]]
            tmp = tmp[:,:,int(tmp.shape[2]/2),int(tmp.shape[3]/2)]

            if index_selected_var is not None:
                tmp = tmp[:,index_selected_var]

            xys = fdata[fdata.files[1]]
            if X_pred is None:
                X_pred = tmp
                coords_pred = xys
            else:
                X_pred = np.concatenate((X_pred, tmp),axis=0)
                coords_pred = np.concatenate((coords_pred, xys),axis=0)

    ## remove duplicates (rows)
    print('X_pred shape before removing duplicates:', X_pred.shape, coords_pred.shape)
    tmp = np.unique(np.concatenate((coords_pred, X_pred), axis=1), axis=0)
    coords_pred = tmp[:,0:2]
    X_pred = tmp[:,2:]
    print('X_pred shape after removing duplicates:', X_pred.shape, coords_pred.shape)

    ## standardize with mean and std per covariate
    data = X_pred
    if not preprocessed and cov_stats_fn is not None:
        stats = np.loadtxt(cov_stats_fn, delimiter=',')

        if index_selected_var is not None:
            stats = stats[index_selected_var,:]

        means = stats[:,2]
        stds = stats[:,3]
        data = (data - means) / stds

    if not preprocessed:
        ## fill nans with column (covariate) means
        col_mean = np.nanmean(data, axis=0)
        #Find indices that you need to replace
        inds = np.where(np.isnan(data))
        #Place column means in the indices. Align the arrays using take
        data[inds] = np.take(col_mean, inds[1])

    return data, coords_pred

def read_data_prediction_cnn(prediction_pnts_fns, nbrhd_size=5, preprocessed=False, cov_stats_fn=None, cov_config_fn=None):
    '''Read in features at prediction points from data files
       Data points are in neighborhood (5 x 5 by default)
    '''
    try:
        X_pred = None
        coords_pred = None
        index_selected_var = None

        if cov_config_fn is not None:
            config = covariates_config.covariates_config(cov_config_fn)
            index_selected_var = config.get_selected_index()

        for prediction_pnts_fn in prediction_pnts_fns:
            ## load features at prediction locations from compressed file
            with np.load(prediction_pnts_fn) as fdata:
                tmp = fdata[fdata.files[0]]

                if tmp.shape[2] < nbrhd_size or tmp.shape[3] < nbrhd_size:
                    print('Neighborhood size is larger than offered in data files. Exiting...')
                    sys.exit(1)
                else:
                    center_x = int(tmp.shape[2]/2)
                    center_y = int(tmp.shape[3]/2)
                    wd = int(nbrhd_size/2)
                    tmp = tmp[:, :, center_x-wd:center_x+wd+1, center_y-wd:center_y+wd+1]

                if index_selected_var is not None:
                    tmp = tmp[:,index_selected_var,:,:]

                xys = fdata[fdata.files[1]]
                if X_pred is None:
                    X_pred = tmp
                    coords_pred = xys
                else:
                    X_pred = np.concatenate((X_pred, tmp),axis=0)
                    coords_pred = np.concatenate((coords_pred, xys),axis=0)

        ## remove duplicates (rows)
        print('X_pred shape before removing duplicates:', X_pred.shape, coords_pred.shape)
        xs = []
        ys = []
        for i in range(X_pred.shape[0]):
            xs.append(np.ones((1, X_pred.shape[2], X_pred.shape[3]))*coords_pred[i,0])
            ys.append(np.ones((1, X_pred.shape[2], X_pred.shape[3]))*coords_pred[i,1])
        xs = np.array(xs)
        ys = np.array(ys)
        tmp = np.unique(np.concatenate((xs, ys, X_pred), axis=1), axis=0)
        coords = tmp[:,0:2,0,0]
        X = tmp[:,2:,:,:]
        print('X_pred shape after removing duplicates:', X.shape, coords.shape)

        ## standardize with mean and std per covariate
        if not preprocessed and cov_stats_fn is not None:
            stats = np.loadtxt(cov_stats_fn, delimiter=',')

            if index_selected_var is not None:
                stats = stats[index_selected_var,:]

            means = []
            stds = []
            for i in range(X.shape[1]):
                means.append(np.ones((X.shape[2],X.shape[3]))*stats[i,2])
                stds.append(np.ones((X.shape[2],X.shape[3]))*stats[i,3])
            means = np.array(means)
            stds = np.array(stds)
            X = (X - means)/stds

        if not preprocessed:
            ## fill nans with column (covariate) means
            col_mean = np.nanmean(X, axis=0)
            #Find indices that you need to replace
            inds = np.where(np.isnan(X))
            #Place column means in the indices. Align the arrays using take
            X[inds] = np.take(col_mean, inds[1])

        return coords, X

    except (Exception) as error:
        print(error)
        raise error

def get_data_iterator(data, batch_size = 10):
    '''Return data iterators over mxnet NDArray
       Example: https://mxnet.apache.org/versions/1.1.0/tutorials/basic/data.html
       data is two-dimensional (center of neighborhood)
    '''
    return mx.io.NDArrayIter(data=data[:,0:-1], label=data[:,-1], batch_size=batch_size, shuffle=True)

def get_data_iterator_cnn(data, batch_size = 10):
    '''Return data iterators over mxnet NDArray
       Example: https://mxnet.apache.org/versions/1.1.0/tutorials/basic/data.html
       data is four-dimensional (5x5 neighborhood)
    '''
    if data.shape[2] == 1 and data.shape[3] == 1:
        #tmp = data.reshape((data.shape[0], data.shape[1]))
        return mx.io.NDArrayIter(data=data[:,0:-1,0,0], label=data[:,-1,0,0], batch_size=batch_size, shuffle=True)
    else:
        return mx.io.NDArrayIter(data=data[:,0:-1,:,:], label=data[:,-1,0,0], batch_size=batch_size, shuffle=True)

def accuracy(y_hat, y):
    ''' Called in mxnet nn/cnn models
        Borrowed from d2l (companion code for Dive into Deep Learning book)
    '''
    #print(y_hat.shape, y.shape)
    if y_hat.shape[1] == 2: # 2 output
        return (y_hat.argmax(axis=1) == y.astype('float32')).sum().asscalar()
    else: # 1 output
        return ((np.squeeze(y_hat) >= 0.5) == y.astype('float32')).sum().asscalar()

def accuracy_model(y_hat, y):
    ''' Called in scikit-learn models, e.g., logistic regression
    '''
    return (y_hat.argmax(axis=1) == y.astype('float32')).sum()

#Evaluate the accuracy for model net on the data set (accessed via data_iter)
def evaluate_accuracy(net, data_iter):
    ''' Called in mxnet nn/cnn models
        Borrowed from d2l (companion code for Dive into Deep Learning book)
    '''
    metric = Accumulator(2) # num_corrected_examples, num_examples
    for batch in data_iter:
        X, y = batch.data[0], batch.label[0]
        metric.add(accuracy(net(X), y), y.size)
    return metric[0] / metric[1]

def evaluate_accuracy_model(model, data_iter):
    ''' Called in scikit-learn models, e.g., logistic regression
    '''
    metric = Accumulator(2) # num_corrected_examples, num_examples
    for batch in data_iter:
        X, y = batch.data[0].asnumpy(), batch.label[0].asnumpy()
        metric.add(accuracy_model(model.predict_proba(X), y), y.size)
    return metric[0] / metric[1]

#Evaluate the accuracy for model net on the data set (accessed via data_iter)
def evaluate_auc(net, data_iter):
    ''' Called in mxnet nn/cnn models
    '''
    metric = Accumulator(1) # auc
    cnt = 0
    for batch in data_iter:
        X, y = batch.data[0], batch.label[0]
        metric.add(scikit_roc_auc(net(X), y))
        cnt += 1
    return metric[0]/cnt

def evaluate_auc_model(model, data_iter):
    ''' Called in scikit-learn models, e.g., logistic regression
    '''
    metric = Accumulator(1) # auc
    cnt = 0
    for batch in data_iter:
        X, y = batch.data[0].asnumpy(), batch.label[0].asnumpy()
        metric.add(scikit_roc_auc_model(model.predict_proba(X), y))
        cnt += 1
    return metric[0]/cnt

def roc_auc_mxndarray(y_hat, y):
    ''' Implement roc_auc() using mnxet.ndarrays
        So that it can be set as a loss function for training mxnet nn/cnn models
        i.e., mxnet.autograd can be used

        !!!THIS IS TO BE IMPLEMENTED!!!
    '''
    yh = y_hat[:,1]
    thresholds = np.arange(0.1, 1.0, 0.1)
    tprs = nd.zeros(thresholds.size)
    fprs = nd.zeros(thresholds.size)
    idx = 0
    for t in thresholds:
        tp = nd.sum((y >= t) * (yh >= t))
        tn = nd.sum((y < t) * (yh < t))
        fp = nd.sum((y < t) * (yh >= t))
        fn = nd.sum((y >= t) * (yh < t))
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        tprs[idx] = tpr
        fprs[idx] = fpr
        idx += 1

    return (tprs * fprs).sum()

def scikit_roc_auc(y_hat, y):
    ''' Called in mxnet nn/cnn models
        It is based on sklearn.metrics.roc_auc_score()
        which takes numpy arrays (NOT mxnet NDArrays) as inputs
        Implication #1: It cannot be used as a loss function for training mxnet nn/cnn models
        Implication #2: If y, y_hat are in gpu memory (e.g., NDArrays predicted from nn/cnn models trained on gpus),
                        both y and y_hat are copied from gpu to cpu (which may be time consuming!!).
    '''
    #print(y_hat.shape, y.shape)
    if y_hat.shape[1] == 2: # 2 output
        return roc_auc_score(y.asnumpy(), y_hat[:,1].asnumpy())
    else: # 1 output
        return roc_auc_score(y.asnumpy(), y_hat.asnumpy())

def scikit_roc_auc_model(y_hat, y):
    ''' Called in scikit-learn models, e.g., logistic regression
    '''
    return roc_auc_score(y, y_hat[:,1])

#Accumulator is a utility class to accumulated sum over multiple numbers.
class Accumulator(object):
    '''Sum a list of numbers over time
       Borrowed from d2l (companion code for Dive into Deep Learning book)
    '''
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a+b for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0] * len(self.data)
    def __getitem__(self, i):
        return self.data[i]

#
def train_epoch(net, train_iter, loss, updater):
    '''training net in one epoch
       Borrowed from d2l (companion code for Dive into Deep Learning book)
    '''
    metric = Accumulator(4) # train_loss_sum, train_acc_sum, num_examples
    if isinstance(updater, gluon.Trainer):
        updater = updater.step
    #for X, y in train_iter:
    cnt = 0
    for batch in train_iter:
        #print('iter # %d' % cnt)
        X, y = batch.data[0], batch.label[0]
        # compute gradients and update parameters
        with autograd.record():
            y_hat = net(X)
            l = loss(y_hat, y)
        l.backward()
        updater(X.shape[0])
        #metric.add(l.sum().asscalar(), accuracy(y_hat, y), y.size)
        #print(scikit_roc_auc(y_hat, y))
        metric.add(l.sum().asscalar(), accuracy(y_hat, y), y.size, scikit_roc_auc(y_hat, y))
        cnt += 1
    # Return training loss and training accuracy
    #print(metric[0])
    #return metric[0]/metric[2], metric[1]/metric[2]
    return metric[0]/metric[2], metric[1]/metric[2], metric[3]/cnt

def use_svg_display():
    """Use the svg format to display plot in jupyter.
       Borrowed from d2l (companion code for Dive into Deep Learning book)
    """
    display.set_matplotlib_formats('svg')

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """A utility function to set matplotlib axes
       Borrowed from d2l (companion code for Dive into Deep Learning book)
    """
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend: axes.legend(legend)
    axes.grid()

class Animator(object):
    '''A utility class that draw data in animation
       Borrowed from d2l (companion code for Dive into Deep Learning book)
    '''
    def __init__(self, xlabel=None, ylabel=None, legend=[], xlim=None,
                 ylim=None, xscale='linear', yscale='linear', fmts=None,
                 nrows=1, ncols=1, figsize=(10, 6)):
        """Incrementally plot multiple lines."""
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1: self.axes = [self.axes,]
        # use a lambda to capture arguments
        self.config_axes = lambda : set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

        self.ylim = ylim

    def add(self, x, y):
        """Add multiple data points into the figure."""
        if not hasattr(y, "__len__"): y = [y]
        n = len(y)
        if not hasattr(x, "__len__"): x = [x] * n
        if not self.X: self.X = [[] for _ in range(n)]
        if not self.Y: self.Y = [[] for _ in range(n)]
        if not self.fmts: self.fmts = ['-'] * n
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

# Save to the d2l package.
def train_net(net, train_iter, eval_iter, loss, num_epochs, updater, figfn='output' + os.sep + 'loss.png'):
    ''' Train neural network models
        Borrowed from d2l (companion code for Dive into Deep Learning book)
    '''
    #trains, eval_accs = [], []
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                        ylim=[0.0, 1.0],
                        legend=['train loss', 'train acc', 'train auc', 'eval acc', 'eval auc'])
    stats_final = None
    for epoch in range(num_epochs):
        #print('epoch # %d' % epoch)
        train_metrics = train_epoch(net, train_iter, loss, updater)
        train_iter.hard_reset()

        eval_acc = evaluate_accuracy(net, eval_iter)
        eval_iter.hard_reset()

        eval_auc = evaluate_auc(net, eval_iter)
        eval_iter.hard_reset()

        #if epoch % 10 == 0:
        print('epoch:%d loss:%.3f train_acc:%.3f train_auc:%.3f eval_acc:%.3f eval_auc:%.3f'\
              % (epoch, train_metrics[0], train_metrics[1], train_metrics[2], eval_acc, eval_auc))
        animator.add(epoch+1, train_metrics+(eval_acc,) + (eval_auc,))

        if epoch == num_epochs-1:
            stats_final = (train_metrics[1], train_metrics[2], eval_acc, eval_auc)

    animator.axes[0].text(num_epochs/2, animator.ylim[0] + 0.1,
                      'train_acc: %.3f train_auc: %.3f\n eval_acc: %.3f eval_auc: %.3f' % stats_final,
                      style='italic',
                      bbox={'facecolor': 'white', 'alpha': 0.2, 'pad': 5})
    animator.fig.savefig(figfn,dpi=300)

# training model in one epoch
def train_model_epoch(model, train_iter):
    '''Training model (e.g., logistic regression in Scikit-learn) in one epoch
       Mimicing train_epoch()
    '''
    metric = Accumulator(3) # train_acc_sum, num_examples, train_auc
    cnt = 0
    for batch in train_iter:
        X, y = batch.data[0].asnumpy(), batch.label[0].asnumpy()
        model.fit(X, y)
        y_hat = model.predict_proba(X)
        metric.add(accuracy_model(y_hat, y), y.size, scikit_roc_auc(nd.array(y_hat), nd.array(y)))
        cnt += 1
    return metric[0]/metric[1], metric[2]/cnt

# Save to the d2l package.
def train_model(model, train_iter, eval_iter, num_epochs, figfn='output' + os.sep + 'model.png'):
    '''Training model (e.g., logistic regression in Scikit-learn)
       Mimicing train_net()
    '''
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                        ylim=[0.3, 1.0],
                        legend=['train acc', 'train auc', 'eval acc', 'eval auc'])

    stats_final = None
    for epoch in range(num_epochs):
        #print('epoch # %d' % epoch)
        train_metrics = train_model_epoch(model, train_iter)
        #print('epoch # %d' % epoch, 'trai acc %.3f' % train_metrics[0], 'trai auc %.3f' % train_metrics[1])
        train_iter.hard_reset()

        eval_acc = evaluate_accuracy_model(model, eval_iter)
        eval_iter.hard_reset()

        eval_auc = evaluate_auc_model(model, eval_iter)
        eval_iter.hard_reset()

        #if epoch % 10 == 0:
        print('epoch:%d train_acc:%.3f train_auc:%.3f eval_acc:%.3f eval_auc:%.3f'\
              % (epoch, train_metrics[0], train_metrics[1], eval_acc, eval_auc))
        animator.add(epoch+1, train_metrics + (eval_acc,) + (eval_auc,))
        if epoch == num_epochs-1:
            stats_final = (train_metrics[0], train_metrics[1], eval_acc, eval_auc)
    animator.axes[0].text(num_epochs/2, animator.ylim[0] + 0.1,
                          'train_acc: %.3f train_auc: %.3f\n eval_acc: %.3f eval_auc: %.3f' % stats_final,
                          style='italic',
                          bbox={'facecolor': 'white', 'alpha': 0.2, 'pad': 5})

    animator.fig.savefig(figfn,dpi=300)

def squared_loss(y_hat, y):
    ''' Squared loss (l2) function
        Borrowed from d2l (companion code for Dive into Deep Learning book)
    '''
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def cross_entropy(y_hat, y):
    ''' Cross_entropy loss function (mxnet has built in CrossEntropy loss function)
        Borrowed from d2l (companion code for Dive into Deep Learning book)
    '''
    return - nd.pick(y_hat, y).log()

def create_raster_from_points(vals, xys, raster_template, out_raster, factor=1.0):
    ''' create a GIS raster layer from point predictions
    '''
    raster_reader = gdalwrapper.tiledRasterReader(raster_template)

    _geotransform = raster_reader.geotransform
    geotransform = (_geotransform[0],
                    _geotransform[1]*factor,
                    _geotransform[2],
                    _geotransform[3],
                    _geotransform[4],
                    _geotransform[5]*factor)

    nrows = int(raster_reader.nrows/factor)
    ncols = int(raster_reader.ncols/factor)

    cellsize = geotransform[1]
    xllcorner = geotransform[0]
    yllcorner = geotransform[3] - cellsize * nrows
    #print('xllcorner %.2f, yllcorner %.2f' % (xllcorner, yllcorner))

    projection = raster_reader.projection

    #print('geotransform:', geotransform)
    #if projection is '':
    #   print('missing projection. use WGS84 by default')

    nodata = -9999.0 #raster_reader.nodata
    #print('nodata:', nodata)

    coords = np.copy(xys)
    coords[:,1] = ((xys[:,0] - xllcorner)/cellsize).astype('int')
    coords[:,0] = nrows - 1 - ((xys[:,1] - yllcorner)/cellsize).astype('int')
    coords = coords.astype(int)

    _coords = []
    _data = []

    for rc, val in zip(coords, vals):
        _coords.append((rc[0], rc[1]))
        _data.append(val)

    if (0, 0) not in _coords:
        _coords.append((0,0))
        _data.append(0)
    if (nrows-1, ncols-1) not in _coords:
        _coords.append((nrows-1, ncols-1))
        _data.append(0)

    coords = np.array(_coords)
    vals = np.array(_data)

    combo = np.concatenate((coords, vals.reshape((coords.shape[0], 1))), axis=1)
    print('combo.shape', combo.shape)
    combo = np.unique(combo, axis=0)
    #print('combo.shape:', combo.shape)
    #print(combo[0:2])

    ## make sure all data points are within the geographic extent
    idx = (combo[:,0] >= 0) * (combo[:,0] <= nrows-1) * (combo[:,1] >= 0) * (combo[:,1] <= ncols-1)
    combo = combo[idx]

    coords = combo[:,0:2].T.astype(int)
    #print('coords.min', np.min(coords, axis=1))
    #print('coords.max', np.max(coords, axis=1))

    vals = combo[:,2]
    #print('vals.min', np.min(vals))
    #print('vals.max', np.max(vals))

    vals = combo[:,2]
    vals[vals==0] = 0.0001

    #print('inside:', np.max(coords, axis=1), np.min(coords, axis=1))

    x = sparse.COO(coords, vals)

    data = x.todense()

    #print('# non-zeros:', np.sum(data != 0))

    data[data==0]=nodata
    #print('data.min', np.min(data))
    #print('data.max', np.max(data))


    print('done writing raster %s. nrows:%d ncols:%d' % (out_raster, nrows, ncols), 'data.shape:', data.shape)
    raster_writer = gdalwrapper.tiledRasterWriter(out_raster, nrows, ncols, 1, geotransform, projection, nodata)
    raster_writer.WriteWholeRaster(data)
    raster_writer.close()

    raster_reader.close()


def fillnan_select_standardize_covariates(data, nodata_list, stats_var= None,index_selected_var=None):
    ''' fill nans, select and standardize variables
    '''
    try:
        #print(nodata_list.shape, stats_var.shape, index_selected_var.shape)
        ## fill nan
        if stats_var is None: # filled with zeros
            for i in range(data.shape[0]):
                data[i][np.isnan(data[i])] = 0
        else: # filled with means
            for i in range(data.shape[0]):
                data[i][np.isnan(data[i])] = stats_var[i,2]

        if stats_var is None and index_selected_var is None:
            return data

        elif stats_var is None and index_selected_var is not None:
            return data[index_selected_var,:,:]

        elif stats_var is not None and index_selected_var is None:
            pass

        else: # stats is not None and index_selected_var is None
            stats_var = stats_var[index_selected_var,:]
            data = data[index_selected_var,:,:]
            nodata_list = nodata_list[index_selected_var]

        ## standardize
        t0 = time.time()
        means = []
        stds = []
        nds = []
        for i in range(data.shape[0]):
            means.append(np.ones((data.shape[1],data.shape[2]))*stats_var[i,2])
            stds.append(np.ones((data.shape[1],data.shape[2]))*stats_var[i,3])

            ## extremely small nodata values (-3.4e+38) are causing troubles
            ## let's replace them
            if nodata_list[i] < -32768:
                data[i][data[i] < -32768] = -32768
                nodata_list[i] = -32768
                #print('nodata changed to -32768')
            nds.append(np.ones((data.shape[1],data.shape[2]))*nodata_list[i])

        means = np.array(means)
        stds = np.array(stds)
        nds = np.array(nds)
        positions = (data != nds)
        #print(positions.shape, positions.sum())
        data[positions] = (data[positions] - means[positions])/stds[positions]
        print('\tstandardizing took %.3f seconds' % (time.time()-t0))
        return data, nodata_list

    except (Exception) as error:
        print(error)
        raise error

def predic_tile(pred_X_tile, nodata_list, nrows, ncols, xoff, yoff, xsize, ysize, nbrhd_size, model, mtype, pred_y):
    ''' apply the model on pre_X_tile for prediction
    '''
    try:
        offset = int(nbrhd_size/2)
        x_start, y_start = 0, 0
        x_end, y_end = xsize, ysize

        if xoff == 0:
            x_start = offset

        if yoff + ysize > nrows-1:
            y_end = ysize - offset - 2

        if yoff == 0:
            y_start = offset

        if xoff + xsize > ncols-1:
            x_end = xsize - offset - 2

        nds = []
        for i in range(pred_X_tile.shape[0]):
            nds.append(np.ones((nbrhd_size,nbrhd_size))*nodata_list[i])
        nds = np.array(nds)
        #print('nds.shape:', nds.shape)
        #print('x_start %d y_start %d' % (x_start, y_start))
        #print('x_end %d y_end %d' % (x_end, y_end))
        t0_total = 0
        t1_total = 0

        coords = []
        data = []

        for row in range(y_start, y_end):
            for col in range(x_start, x_end):
        #for row in range(ysize):
        #    for col in range(xsize):
                #'''
                #print('working on row %d col %d' % (row, col))
                #X = pred_X_tile[:, row : row + nbrhd_size, col + nbrhd_size]
                t0 = time.time()
                if y_start == 0:
                    if x_start == 0:
                        X = pred_X_tile[:, row : row + nbrhd_size, col : col + nbrhd_size]
                    else: # x_start == offset
                        X = pred_X_tile[:, row : row + nbrhd_size, col - offset : col + offset + 1]
                else: #y_start == offset
                    if x_start == 0:
                        X = pred_X_tile[:, row - offset : row + offset + 1, col : col + nbrhd_size]
                    else: #x_start == offset
                        X = pred_X_tile[:, row - offset : row + offset + 1, col - offset : col + offset + 1]

                if X.shape != nds.shape:
                    print('%d %d' % (row, col))
                    print(pred_X_tile.shape, pred_y.shape)
                    print(X.shape, nds.shape)
                    sys.exit(0)
                positions = (X == nds)
                #print('nds[0]:', nds[0])
                #print('X[0]: ', X[0])
                #print('positions[0]: ', positions[0])
                #print('positions.sum(): ', positions.sum())
                #sys.exit(0)
                t0_total += time.time()-t0
                #print('inside: # of nodata ', positions.sum())
                if positions.sum() > 0:
                    continue
                else:
                    #print('prediction at row %d col %d' % (row, col))
                    #val = random.random()
                    data.append(X)
                    ## col, row in the tile (local)
                    coords.append([col, row])
                    #pred_y[row, col] = np.mean(X)
                #t1_total += time.time()-t1
                #'''
                #pred_y[row, col] = random.random()
        #if mtype == 'net': pass
        #elif mtype == 'scikit': pass

        data = np.array(data)
        coords = np.array(coords)
        #print('inside: ', data.shape, coords.shape)

        if data.size > 0:
            ## do model prediction here
            t1 = time.time()
            if mtype == 'net':
                #data = nd.array(data)
                #if model[0].weight.data().context == data.context:
                #    _y_pred = model(data).asnumpy()
                #else:
                _y_pred = model(nd.array(data).as_in_context(model[0].weight.data().context)).asnumpy()
                # output prediction in raster format
                if _y_pred.shape[1]==2:
                    _y_pred = _y_pred[:,1]
                else:
                    _y_pred = np.squeeze(_y_pred)

            elif mtype == 'scikit':

                if len(data.shape) != 2:
                    data = data.reshape(data.shape[0], data.shape[1])
                _y_pred = model.predict_proba(data)[:,1]

            else:
                _y_pred = np.mean(np.mean(np.mean(data, axis=3), axis=2), axis=1)

            #print(type(_y_pred))
            #print(_y_pred)
            pred_y[coords[:,1], coords[:,0]] = _y_pred
            t1_total = time.time()-t1

            print('\tdata prep took %.3f seconds' % t0_total)
            print('\tprediction took %.3f seconds' % t1_total)

            ## col, row in the whole raster map (global)
            coords[:,0] = coords[:,0] + xoff
            coords[:,1] = coords[:,1] + yoff

        return coords, data

    except (Exception) as error:
        print(error)
        raise error

def predict_map(cov_base_dir=None, cov_group_fns=None, nbrhd_size=5, preprocessed=False, cov_stats_fn=None, cov_config_fn=None, \
                model=None, mtype='net', out_prediction_map_fn=None, write_files=False, out_cov_base_dir=None, out_feature_dir=None):
    ''' read raster stack and apply the model pixel by pixel to produce a map
    '''
    if cov_base_dir is None:
        cov_base_dir = 'D:/OneDrive - University of Denver/eBird/covariates/americas'

    if cov_group_fns is None:
        cov_group_fns = ['bioclimatic_variables.vrt',\
               'global_habitat_heterogeneity.vrt',\
               'gpw_pop_density.vrt',\
               'grip4_road_density.vrt',\
               'landcover_prevalence.vrt',\
               'topo_cont_vars_max.vrt',\
               'topo_cont_vars_mean.vrt',\
               'topo_cont_vars_median.vrt',\
               'topo_cont_vars_min.vrt',\
               'topo_cont_vars_std.vrt',\
               'topo_landform_class_pcnt.vrt']

    #out_map_template = 'output/map_template.tif'
    if out_prediction_map_fn is None:
        out_prediction_map_fn = 'output/map_prediction_test.tif'
    NODATA = -9999.0

    overlap = int(nbrhd_size/2)

    try:

        index_selected_var = None
        if cov_config_fn is not None:
            config = covariates_config.covariates_config(cov_config_fn)
            index_selected_var = config.get_selected_index()
            #config.print_variables()

        stats_var = None
        if cov_stats_fn is not None:
            stats_var = np.loadtxt(cov_stats_fn, delimiter=',')

        nodata_list = []
        nbands_list = []
        readers = []
        for cov_fn in cov_group_fns:
            cov_fn = cov_base_dir + os.sep + cov_fn
            print("Covariate: %s" % cov_fn)
            reader = gdalwrapper.tiledRasterReader(cov_fn)
            # assuming all bands share the same NoDataValue

            nbands_list.append(reader.nbands)

            for i in range(reader.nbands):
                nodata_list.append(reader.nodatas[i])
            '''
            reader.xoff = 0
            reader.yoff = 16600
            reader.setNTilesRead(166)
            '''

            readers.append(reader)

        nodata_list = np.array(nodata_list)

        COORDS = None
        DATA = None
        projection = readers[0].projection
        geotransform = readers[0].geotransform
        nrows = readers[0].nrows
        ncols = readers[0].ncols
        cellsize = geotransform[1]
        xllcorner = geotransform[0]
        yllcorner = geotransform[3] - cellsize * nrows

        ## output raster - prediction
        writer = gdalwrapper.tiledRasterWriter(out_prediction_map_fn, nrows, ncols,
                                               1, geotransform, projection, NODATA)

        ## output rasters - individual covariates (standardized, nans filled)
        if write_files:
            if out_cov_base_dir is None:
                out_cov_base_dir = 'D:/OneDrive - University of Denver/eBird/covariates/americas/individual_variables'
            writers_covs = []
            for idx in index_selected_var:
                cov_fn = config.variables[idx].split('=')[0].replace(' ','') + '.tif'
                #cov_fn = 'test.tif'
                out_cov_fn = out_cov_base_dir + os.sep + cov_fn
                print(out_cov_fn)
                writers_covs.append(gdalwrapper.tiledRasterWriter(out_cov_fn, nrows, ncols, 1, geotransform, projection, NODATA))

        ## set tile size
        if '.vrt' in cov_group_fns[0]:
            xsize = readers[0].xsize
            ysize = readers[0].ysize# / 2
        else: # '.tif'
            xsize = readers[0].ysize
            ysize = readers[0].xsize * 100
        print(xsize, ysize)

        cnt = 0
        ## first tile
        pred_X_tile = None
        t00 = time.time()
        data = None
        xoff, yoff = 0, 0
        for reader in readers:
            data, xoff, yoff, xsize, ysize, _xoff, _yoff = reader.readNextTileOverlap(xsize = xsize, ysize = ysize, overlap = overlap)
            #print(data.shape, xoff, yoff, xsize, ysize)

            if pred_X_tile is None:
                pred_X_tile = data
            else:
                pred_X_tile = np.concatenate((pred_X_tile, data),axis=0)

        if pred_X_tile is not None:
            print(pred_X_tile.shape)
            print('reading tile %d %d took %.3f seconds' % (xoff, yoff, time.time()-t00))

            while data is not None:# and cnt < 2:
                if not preprocessed:
                    tx = time.time()
                    pred_X_tile, nd = fillnan_select_standardize_covariates(pred_X_tile, np.copy(nodata_list), np.copy(stats_var),\
                                                                            index_selected_var)
                    print('post-processing tile %d %d took %.3f seconds' % (xoff, yoff, time.time()-tx))
                else:
                    nd = np.copy(nodata_list)

                ## write tiles to raster file by covariate
                if write_files:
                    tx = time.time()
                    for _X, _nd, _writer_cov in zip(pred_X_tile, nd, writers_covs):
                        _data = np.copy(_X)
                        _data[_data==_nd] = NODATA
                        #print(_nd, np.min(_data[_data != NODATA]), (np.max(_data[_data != NODATA])))
                        _writer_cov.writeTile(_data, _xoff, _yoff)
                    print('...writing tile %d %d to file took %.3f seconds' % (xoff, yoff, time.time()-tx))

                ## do prediction on this first tile
                ## TBD
                pred_y = np.ones((ysize, xsize)) * NODATA
                tx = time.time()
                _COORDS, _DATA = predic_tile(pred_X_tile, nd, nrows, ncols, xoff, yoff, xsize, ysize, nbrhd_size, model, mtype, pred_y)
                print('predicting tile %d %d took %.3f seconds' % (xoff, yoff, time.time()-tx))
                
                '''
                if _COORDS.size > 0 and write_files:
                    ## write features data to .npz file
                    #print('nrows %d, ncols %d, cellsize %.7f, xllcorner %.7f, yllcorner %.7f' % (nrows, ncols, cellsize, xllcorner,\
                    #                                                                              yllcorner))
                    if out_feature_dir is None:
                        out_feature_dir = 'data' + os.sep + 'standardized'
                    #print(_COORDS)
                    tx = time.time()
                    _COORDS = _COORDS.astype(np.float32)
                    _COORDS[:,0] = _COORDS[:,0] * cellsize + xllcorner
                    _COORDS[:,1] = (nrows - 1 - _COORDS[:,1]) * cellsize + yllcorner
                    #print(_COORDS)
                    np.savez_compressed(out_feature_dir + os.sep + 'americas_tile_' + str(xoff)+ '_' +str(yoff) + '.npz',
                                        data = _DATA, coords = _COORDS)
                    print('...writing features data in tile %d %d took %.3f seconds' % (xoff, yoff, time.time()-tx))
                '''
                ## write predicted tile
                tx = time.time()
                writer.writeTile(pred_y, xoff, yoff)
                print('writing prediction tile %d %d took %.3f seconds' % (xoff, yoff, time.time()-tx))

                cnt += 1

                ## proceed to next tile
                pred_X_tile = None
                t0 = time.time()
                for reader in readers:
                    data, xoff, yoff, xsize, ysize, _xoff, _yoff = reader.readNextTileOverlap(overlap = overlap)
                    #print(data.shape, xoff, yoff, xsize, ysize)
                    if pred_X_tile is None:
                        pred_X_tile = data
                    else:
                        pred_X_tile = np.concatenate((pred_X_tile, data),axis=0)

                if pred_X_tile is not None:
                    print(pred_X_tile.shape)
                    print('reading tile %d %d took %.3f seconds' % (xoff, yoff, time.time()-t0))

            print('IN TOTAL it took %.3f seconds' % (time.time()-t00))

        tx = time.time()

        writer.close()

        for reader in readers:
            reader.close()

        if write_files:
            for writer_cov in writers_covs:
                writer_cov.close()

        print('cleaning up took %.3f seconds' % (time.time()-tx))


    except (Exception) as error:
        print(error)
        raise error


def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    return context.gpu(i) if context.num_gpus() >= i + 1 else context.cpu()

# Save to the d2l package.
def try_all_gpus():
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    ctxes = [context.gpu(i) for i in range(context.num_gpus())]
    return ctxes if ctxes else [context.cpu()]

def train_test_split_with_coords(coords, data, test_size):
    '''Split a dataset into train and test sets 
       based on sklearn.model_selection import train_test_split
       Dataset has both features and labels (in data) and optionally coordinates in coords 
    '''
    try:
        idx = np.array(range(data.shape[0]))
        idx_tr, idx_te = train_test_split(idx, test_size=test_size)
        
        train = data[idx_tr] 
        coords_tr = coords[idx_tr]
        
        test = data[idx_te]
        coords_te = coords[idx_te]
        
        return coords_tr, train, coords_te, test
    
    except (Exception) as error:
        print(error)
        raise error

def coordsToCSV(coords, fn):
    ''' Save coords to CSV file
    '''
    try:
        with open(fn, 'w') as file:
            np.savetxt(file, coords, delimiter=',', header='x,y,label')
            
    except (Exception) as error:
        print(error)
        raise error