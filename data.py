"""
This script contains some utility functions related to data generation' 

We acknowledge the use of existing code from the repositories in https://github.com/wittawatj/ that served as a skeleton for the implementation of some of our tests, especially kernel-based tests. Thank you!
"""

from __future__ import print_function
from __future__ import division
from builtins import map
from builtins import str
from builtins import range
from past.utils import old_div
from builtins import object

import math
import matplotlib.pyplot as plt
import autograd.numpy as np

import scipy.stats as stats
import random
import pandas as pd
from math import pi
import random
import pickle
from scipy.stats import chi2, invgamma
from itertools import compress

def same(x):
    return x

def cube(x):
    return np.power(x, 3)

def negexp(x):
    return np.exp(-np.abs(x))

def tr_te_indices(n, tr_proportion, seed=9282):
    """Get two logical vectors for indexing train/test points.
    Return (tr_ind, te_ind)
    """
    rand_state = np.random.get_state()
    np.random.seed(seed)

    Itr = np.zeros(n, dtype=bool)
    tr_ind = np.random.choice(n, int(tr_proportion * n), replace=False)
    Itr[tr_ind] = True
    Ite = np.logical_not(Itr)

    np.random.set_state(rand_state)
    return (Itr, Ite)

class TSTData(object):
    """Class representing data for two-sample test"""

    """
    properties:
    X, Y: numpy array 
    """

    def __init__(self, X, Y, wx, wy, label=None):
        """
        :param X: n x d numpy array for dataset X1 with confounders of population
        :param Y: n x d numpy array for dataset Y
        """
        self.X = X
        self.Y = Y
        self.wx = wx
        self.wy = wy
        # short description to be used as a plot label
        self.label = label

        nx, dx = X.shape
        ny, dy = Y.shape

        # if nx != ny:
        #    raise ValueError('Data sizes must be the same.')
        if dx != dy:
            raise ValueError('Dimension sizes of the two datasets must be the same.')

    def __str__(self):
        mean_x = np.mean(self.X, 0)
        std_x = np.std(self.X, 0)
        mean_y = np.mean(self.Y, 0)
        std_y = np.std(self.Y, 0)
        prec = 4
        desc = ''
        desc += 'E[x] = %s \n ' %(np.array_str(mean_x, precision=prec ) )
        desc += 'E[y] = %s \n ' %(np.array_str(mean_y, precision=prec ) )
        desc += 'Std[x] = %s \n' %(np.array_str(std_x, precision=prec))
        desc += 'Std[y] = %s \n' % (np.array_str(std_y, precision=prec))
        return desc

    def dimension(self):
        """Return the dimension of the data."""
        dx = self.X.shape[1]
        return dx

    def dim(self):
        """Same as dimension()"""
        return self.dimension()

    def stack_xy(self):
        """Stack the two datasets together"""
        return np.vstack((self.X, self.Y))

    def xy(self):
        """Return (X, Y) as a tuple"""
        return (self.X, self.Y)
    
    def weights(self):
        """Return weights as a tuple"""
        return (self.wx, self.wy)

    def mean_std(self):
        """Compute the average standard deviation """

        # Gaussian width = mean of stds of all dimensions
        X, Y = self.xy()
        stdx = np.mean(np.std(X, 0))
        stdy = np.mean(np.std(Y, 0))
        mstd = old_div((stdx + stdy), 2.0)
        return mstd
        # xy = self.stack_xy()
        # return np.mean(np.std(xy, 0)**2.0)**0.5

    def split_tr_te(self, tr_proportion=0.5, seed=820):
        """Split the dataset into training and test sets. Assume n is the same
        for both X, Y.

        Return (TSTData for tr, TSTData for te)"""
        X = self.X
        Y = self.Y
        wx, wy = self.wx, self.wy
        nx, dx = X.shape
        ny, dy = Y.shape
        if nx != ny:
            raise ValueError('Require nx = ny')
        Itr, Ite = tr_te_indices(nx, tr_proportion, seed)
        wxtr, wytr = list(compress(wx, Itr)), list(compress(wy, Itr))
        wxte, wyte = list(compress(wx, Ite)), list(compress(wy, Ite))
        label = '' if self.label is None else self.label
        tr_data = TSTData(X[Itr, :], Y[Itr, :], wxtr, wytr, 'tr_' + label)
        te_data = TSTData(X[Ite, :], Y[Ite, :], wxte, wyte, 'te_' + label)
        return (tr_data, te_data)

    def subsample(self, n, seed=87):
        """Subsample without replacement. Return a new TSTData """
        if n > self.X.shape[0] or n > self.Y.shape[0]:
            raise ValueError('n should not be larger than sizes of X, Y.')
        ind_x = subsample_ind(self.X.shape[0], n, seed)
        ind_y = subsample_ind(self.Y.shape[0], n, seed)
        return TSTData(self.X[ind_x, :], self.Y[ind_y, :], self.label)
    

'''
--------------------------------------------------------------------------------------------
Generate two samples of random functions observed at irregularly sampled times
--------------------------------------------------------------------------------------------
'''

def generate_random_functions(size=1000, num_obs = 100, function= 'sine',mean_scale = 1, variance = 0.1, error='gaussian',mean=0,meta_mu=7):
    '''Generate samples from random functions
    Arguments:
        size : number of samples
        nun_obs: number of observations in each trajectory
        function: specify function for the mean trend
        variance: error variance
        erorr: distribution of the error, one of 'gaussian', 'laplace' or 'exponential'
    Output:
        array of times and array of observations at those times
    '''

    np.random.seed()

    #Y = np.zeros(num_obs) # placeholder for observation values
    #T = np.zeros(num_obs) # placeholder for observation times
    Y = []
    T = []
    for i in range(size):
        num_obs = int(np.random.uniform(low=10,high=50,size=1))
        t = np.random.uniform(size=num_obs,low=0,high=1)
        t = np.sort(t)

        if function == 'sine':
            mu = mean_scale * np.sin(2*pi*t) + mean
        if function == 'zero':
            mu = np.zeros(len(t))
        if function == 'spike':
            mu = mean_scale * np.exp(-(t-0.5)**2/0.005)

        if error == 'gaussian':
            meta_var = invgamma.rvs(meta_mu,size=1) # meta_mu needs to be set with care
            #cov = np.diag(np.ones(len(t))*(meta_var + variance)
            y = mu + np.random.normal(0, meta_var + variance, num_obs)
        if error == 'laplace':
            variance = invgamma.rvs(meta_mu,size=1) # meta_mu needs to be set with care
            y = mu + np.random.laplace(loc=0,scale = variance, size=len(mu))
        if error == 'exponential':
            variance = invgamma.rvs(meta_mu,size=1) # meta_mu needs to be set with care
            #var_exp = chi2.rvs(df=1, loc=variance, size=1)
            y = mu + np.random.exponential(scale= variance, size=len(mu))
        

        Y.append(y)#Y = np.vstack((Y, y))
        T.append(t)#T = np.vstack((T, t))

    return T, Y#np.array(T[1:]), np.array(Y[1:])


def generate_conditional_functions(size=250, num_obs = 100, function= 'sine',mean_scale = 1, var = 0.1, error='gaussian',mean=0, debug=False,transformation = None, meta_mu=5):
    '''Generate samples for three variables X, Y, Z from random functions either conditionally independent or not
    Arguments:
        size : number of samples
        nun_obs: number of observations in each trajectory
        function: specify function for the mean trend
        variance: error variance
        error: distribution of the error, one of 'gaussian', 'laplace' or 'exponential'
    Output:
        array of times and array of observations at those times for all variables
        [nxd array of T], [nxd array of X], [nxd array of T], [nxd array of Y]
        
    TODO: I should sample very different trajectories for each unit but keep the 
    '''

    np.random.seed()

    X = np.zeros(num_obs); Y = np.zeros(num_obs); # placeholder for observation values
    T = np.zeros(num_obs) # placeholder for observation times, can extend to different observations per variable
    T2 = np.zeros(num_obs)

    if transformation == None: 
        transformation = random.randint(1, 4)

    if transformation == 1:
        f1 = np.square
    elif transformation == 2:
        f1 = cube
    elif transformation == 3:
        f1 = np.cos
    else:
        f1 = negexp

    if debug:
        print(f1)

        
    for i in range(size):
        t = np.random.uniform(size=num_obs,low=0,high=1); t2 = np.random.uniform(size=num_obs,low=0,high=1) 
        #t = np.sort(t);t2 = np.sort(t2)

        mean_scale = np.random.uniform(size=1,low=0.5,high=1.5)
        mean_t = np.random.uniform(size=1,low=- 0.5,high=+ 0.5)
        mean = np.random.uniform(size=1,low=- 0.5,high=+ 0.5)

        if function == 'sine':
            mu = mean_scale * np.sin(2*pi*t) + mean_t * t + mean
        if function == 'zero':
            mu = np.zeros(len(t)) + mean
            mu2 = np.zeros(len(t2)) + mean
        if function == 'spike':
            mu = mean_scale * np.exp(-(t-0.5)**2/0.005)

        if error == 'gaussian':
            #var = invgamma.rvs(5,size=1) 
            x = mu + np.random.normal(0, var, num_obs)
        if error == 'laplace':
            #var = invgamma.rvs(5,size=1) 
            x = mu + np.random.laplace(loc=0,scale = var, size=len(mu))
        if error == 'exponential':
            #var = invgamma.rvs(5,size=1) 
            x = mu + np.random.exponential(scale= var, size=len(mu))
        
        #var = invgamma.rvs(meta_mu,size=1) # meta_mu needs to be set with care
        y = f1(mu) + np.random.normal(0, var, num_obs)

        X = np.vstack((X, x)); Y = np.vstack((Y, y)); 
        T = np.vstack((T, t)); T2 = np.vstack((T2, t2))

    return np.array(T[1:]), np.array(X[1:]), np.array(T[1:]), np.array(Y[1:])


'''
--------------------------------------------------------------------------------------------
Import Cystic Fibrosis Data for analysis
--------------------------------------------------------------------------------------------
'''

def generate_CF_samples(outcome = 'FEV1_PREDICTED', feature = 'Gender'):
    '''
    :param outcome: choose outcome, either FEV1 or FEV1PREDICTED
    :param feature: feature to split the data into two samples
    :return: sets of times and observed values
    '''
    if outcome == 'FEV1':
        with open ('C:/Users/abellot/Documents/Python Scripts/Cystic Fibrosis/FEV1', 'rb') as fp:
            outcome = pickle.load(fp)
    else:
        with open ('C:/Users/abellot/Documents/Python Scripts/Cystic Fibrosis/FEV1_PREDICTED', 'rb') as fp:
            outcome = pickle.load(fp)

    with open ('C:/Users/abellot/Documents/Python Scripts/Cystic Fibrosis/TIMES', 'rb') as fp:
        TIMES = pickle.load(fp)
    with open('C:/Users/abellot/Documents/Python Scripts/Cystic Fibrosis/BASELINE', 'rb') as fp:
        BASELINE = pickle.load(fp)



    idx_0 = [i for i, e in enumerate(BASELINE[feature]) if e == 0]
    idx_1 = [i for i, e in enumerate(BASELINE[feature]) if e == 1]

    t0, t1 = [TIMES[i] for i in idx_0], [TIMES[i]  for i in idx_1]
    y0, y1 = [outcome[i]  for i in idx_0], [outcome[i]  for i in idx_1]
    
    
    def filter_nans(x, y):
        filtered1 = filter(lambda o: not np.isnan(o[0]) and not np.isnan(o[1]), zip(x, y))    
        filtered2 = filter(lambda o: not np.isnan(o[0]) and not np.isnan(o[1]), zip(x, y))    
        return [el[0] for el in filtered1], [el[1] for el in filtered2]
    
    t0_final = []
    y0_final = []
    t1_final = []
    y1_final = []
    
    for times, outcomes in zip(t0,y0):
        temp_t, temp_y = filter_nans(times,outcomes)
        t0_final.append(temp_t)
        y0_final.append(temp_y)
        
    for times, outcomes in zip(t1,y1):
        temp_t, temp_y = filter_nans(times,outcomes)
        t1_final.append(temp_t)
        y1_final.append(temp_y)
    
    t0_final = [[float(j)/2557 for j in i] for i in t0_final]
    t1_final = [[float(j)/2557 for j in i] for i in t1_final]
    
    # remove empty lists
    t0_final = [x for x in t0_final if len(x) != 0 and len(x) != 1] 
    t1_final = [x for x in t1_final if len(x) != 0 and len(x) != 1] 
    y0_final = [x for x in y0_final if len(x) != 0 and len(x) != 1] 
    y1_final = [x for x in y1_final if len(x) != 0 and len(x) != 1] 
    
    # convert to numpy arrays
    #t0_final = np.array([np.array(item) for sublist in t0_final for item in sublist])
    #t1_final = np.array([np.array(item) for sublist in t1_final for item in sublist])
    #y0_final = np.array([np.array(item) for sublist in y0_final for item in sublist])
    #y1_final = np.array([np.array(item) for sublist in y1_final for item in sublist])     
   
    #t0 = np.reshape(t0[~np.isnan(y0)],(-1,1))
    #t1 = np.reshape(t1[~np.isnan(y1)],(-1,1))/2557
    #y0 = np.reshape(y0[~np.isnan(y0)],(-1,1))
    #y1 = np.reshape(y1[~np.isnan(y1)],(-1,1))
    
    return t0_final, y0_final, t1_final, y1_final






