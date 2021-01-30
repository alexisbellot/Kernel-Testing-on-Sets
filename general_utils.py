"""A module containing general utility methods for performance comparisons, data manipulation and
random feature generation"""


from __future__ import print_function
from builtins import object
import autograd.numpy as np
import time
import pandas as pd
import os
import sys
import math
from collections import defaultdict
import joblib
import data as data
import numpy as np
import random
from tqdm import tqdm
from two_sample_test_utils import RMMD


def performance_comparisons(methods,num_runs,param_name,params,alpha=0.05,mean_scale = 1, output='p_value', 
                            verbose=False, size=300, num_obs = 10, var1=.1, var2=.1, function1 = 'sine', 
                            function2 = 'sine',error1="gaussian", error2="gaussian",meta_mu=7):
    '''
    This function computes performance comparison runs
    
    methods: name of methods to be tested
    num_runs: number of run results averaged over
    param: parameter vector to iterate over
    alpha: significance level
    '''
    
    performance = defaultdict(int)

    for param in params:
        
        # define which parameter to iterate over
        if param_name == 'num_obs':
            num_obs = param
        if param_name == 'var1':
            var1 = param
        if param_name == 'size':
            size = param
        if param_name == 'mean_scale':
            mean_scale = param
        if param_name == 'error':
            error = param
                
        for n_run in tqdm(range(num_runs)):
            np.random.seed(n_run)
        
            # Create a null dataset.
            t1, y1 = data.generate_random_functions(size=size, num_obs = num_obs, mean_scale = mean_scale,\
                                                    function= function1,variance = var1, error=error1,\
                                                    meta_mu=meta_mu)
            t2, y2 = data.generate_random_functions(size=size, num_obs = num_obs, mean_scale = 1,\
                                                    function= function2, variance = var2, error=error2,\
                                                    meta_mu=meta_mu)
            
            if n_run % 100 == 0 and verbose == True:
                print('=' * 70)
                print('Sample output for parameter:', param)
                print('=' * 70)

            # Run the tests on both data sets and compute type I and II errors.
            for method in methods:
                method_name = method.__name__
                key2 = 'method: {}; param value: {} '.format(method_name, param)
                tic = time.time()
                pval = method(t1, y1, t2, y2, output=output)
                toc = (time.time() - tic) / 2.
                
                # performance refers to tyoe I error if data generated under the null
                # and referes to power if data under the alternative
                performance[key2] += int(pval < alpha) / num_runs
                

                if n_run % 100 == 0 and verbose == True:
                    print('{}: time={:.2}s, p_value={:.4}.'.format( method_name, toc, pval))
                    
    return performance

def performance_comparisons_indep(methods,num_runs,param_name,params,alpha=0.05,mean_scale = 1, output='p_value', 
                            verbose=False, size=100, num_obs = 10, var=.1, data_type='ind',transformation=None,
                                 meta_mu=5):
    '''
    This function computes performance comparison runs
    
    methods: name of methods to be tested
    num_runs: number of run results averaged over
    param: parameter vector to iterate over
    alpha: significance level
    '''
    
    performance = defaultdict(int)

    for param in params:
        
        # define which parameter to iterate over
        if param_name == 'num_obs':
            num_obs = param
        if param_name == 'size':
            size = param
        if param_name == 'meta_mu':
            meta_mu = param
        if param_name == 'var':
            var = param

                
        for n_run in tqdm(range(num_runs)):
            np.random.seed(n_run)
        
            # Create a dataset under the alternative hypothesis of dependence.
            if data_type == 'ind':
                t1, y1, _, _ = data.generate_conditional_functions(size=size, num_obs = num_obs,
                                                                     function='sine',meta_mu=meta_mu,var=var)
                t2, y2, _, _ = data.generate_conditional_functions(size=size, num_obs = num_obs, 
                                                                   function='zero', meta_mu=meta_mu,var=var)
            if data_type == 'dep':
                t1, y1, t2, y2 = data.generate_conditional_functions(size=size, num_obs = num_obs,
                                                                     function='sine',meta_mu=meta_mu,
                                                                    transformation=transformation,var=var)
            
            if n_run % 100 == 0 and verbose == True:
                print('=' * 70)
                print('Sample output for parameter:', param)
                print('=' * 70)

            # Run the tests on both data sets and compute type I and II errors.
            for method in methods:
                method_name = method.__name__
                key2 = 'method: {}; param value: {} '.format(method_name, param)
                tic = time.time()
                pval = method(t1, y1, t2, y2, output=output)
                toc = (time.time() - tic) / 2.
                
                # performance refers to tyoe I error if data generated under the null
                # and referes to power if data under the alternative
                performance[key2] += int(pval < alpha) / num_runs
                

                if n_run % 100 == 0 and verbose == True:
                    print('{}: time={:.2}s, p_value={:.4}.'.format( method_name, toc, pval))
                    
    return performance


def time_complexity(methods,sizes,num_runs=10, num_obs = 10):
    
    
    times = defaultdict(int)
    for size in tqdm(sizes):
        t1, y1 = data.generate_random_functions(size=size, num_obs = num_obs)
        t2, y2 = data.generate_random_functions(size=size, num_obs = num_obs)
        
        for method in methods:
            method_name = method.__name__
            key = 'method: {}, number of samples {}'.format(method_name, size)
            tic = time.time()
            [method(t1, y1, t2, y2, output='p_value') for i in range(num_runs)]
            toc = (time.time() - tic)
            times[key] = toc / num_runs
        
    return times   
 

def perf_num_features(num_runs, num_features, alpha=0.05,mean_scale = 1, output='p_value', 
                verbose=False, size=300, num_obs = 10, var1=.1, var2=.1, function1 = 'sine', 
                function2 = 'sine',error1="gaussian", error2="gaussian"):
    '''
    This function computes performance as a function of the number of random features used to approximate
    the mean embedding.
    
    num_runs: number of run results averaged over

    '''
    
    performance = defaultdict(int)

    for feat in num_features:
                
        for n_run in tqdm(range(num_runs)):
            np.random.seed(n_run)
        
            # Create a null dataset.
            t1, y1 = data.generate_random_functions(size=size, num_obs = num_obs, mean_scale = mean_scale,\
                                                    function= function1,variance = var1, error=error1)
            t2, y2 = data.generate_random_functions(size=size, num_obs = num_obs, mean_scale = 1,\
                                                    function= function2, variance = var2, error=error2)
            if n_run % 100 == 0 and verbose == True:
                print('=' * 70)
                print('Sample output for parameter:', param)
                print('=' * 70)

            # Run the tests on both data sets and compute type I and II errors.
            
            method_name = 'RMMD'
            key = 'method: {}; number of features: {} '.format(method_name, feat)
            tic = time.time()
            pval = RMMD(t1, y1, t2, y2, output=output)
            toc = (time.time() - tic) / 2.
                
            # performance refers to tyoe I error if data generated under the null
            # and referes to power if data under the alternative
            performance[key] += int(pval < alpha) / num_runs
                
            if n_run % 100 == 0 and verbose == True:
                print('{}: time={:.2}s, p_value={:.4}.'.format( method_name, toc, pval))
                    
    return performance

def constrain(val, min_val, max_val):
    return min(max_val, max(min_val, val))


def dist_matrix(X, Y):
    """
    Construct a pairwise Euclidean distance matrix of size X.shape[0] x Y.shape[0]
    """
    sx = np.sum(X ** 2, 1)
    sy = np.sum(Y ** 2, 1)
    D2 = sx[:, np.newaxis] - 2.0 * np.dot(X, Y.T) + sy[np.newaxis, :]
    # to prevent numerical errors from taking sqrt of negative numbers
    D2[D2 < 0] = 0
    D = np.sqrt(D2)
    return D


def meddistance(X, subsample=None, mean_on_fail=True):
    """
    Compute the median of pairwise distances (not distance squared) of points
    in the matrix.  Useful as a heuristic for setting Gaussian kernel's width.
    Parameters
    ----------
    X : n x d numpy array
    mean_on_fail: True/False. If True, use the mean when the median distance is 0.
        This can happen especially, when the data are discrete e.g., 0/1, and
        there are more slightly more 0 than 1. In this case, the m
    Return
    ------
    median distance
    """
    if subsample is None:
        D = dist_matrix(X, X)
        Itri = np.tril_indices(D.shape[0], -1)
        Tri = D[Itri]
        med = np.median(Tri)
        if med <= 0:
            # use the mean
            return np.mean(Tri)
        return med

    else:
        assert subsample > 0
        rand_state = np.random.get_state()
        np.random.seed(9827)
        n = X.shape[0]
        ind = np.random.choice(n, min(subsample, n), replace=False)
        np.random.set_state(rand_state)
        # recursion just one
        return meddistance(X[ind, :], None, mean_on_fail)


def is_real_num(x):
    """return true if x is a real number"""
    try:
        float(x)
        return not (np.isnan(x) or np.isinf(x))
    except ValueError:
        return False


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


def subsample_ind(n, k, seed=28):
    """
    Return a list of indices to choose k out of n without replacement
    """
    rand_state = np.random.get_state()
    np.random.seed(seed)

    ind = np.random.choice(n, k, replace=False)
    np.random.set_state(rand_state)
    return ind




def rp(k,s,d):
    '''
    This function samples random frequencies from the fourier transform of the gaussian kernel 
    (that is from a gaussian distribution) and uniform samples in [0,2pi]
    param:
    - k = number of random features
    - s = median heuristic for standard deviation of gaussian distribution, s**2 is the variance.
          In original paper of Lopez Paz, three embeddings are created with different scales s = [s1,s2,s3]
    - d = dimensionality of w
    '''
    #s_list = [0.5*s,s,2*s] 
    #w = np.vstack([si*np.random.randn(k,d) for si in s_list])
    #b = 2*np.pi*np.random.rand(3*k,1)
    
    # with one parameter s use the following
    w = s*np.random.randn(k,d)
    b = 2*np.pi*np.random.rand(k,1)
    
    return np.hstack((w,b)).T

def rp_ind(k,s,d):
    '''
    This function samples random frequencies from the fourier transform of the gaussian kernel 
    (that is from a gaussian distribution) and uniform samples in [0,2pi]
    param:
    - k = number of random features
    - s = median heuristic for standard deviation of gaussian distribution, s**2 is the variance.
          In original paper of Lopez Paz, three embeddings are created with different scales s = [s1,s2,s3]
    - d = dimensionality of w
    '''
    s_list = [0.5*s,s,2*s] 
    w = np.vstack([si*np.random.randn(k,d) for si in s_list])
    b = 2*np.pi*np.random.rand(3*k,1)
    
    # with one parameter s use the following
    #w = s*np.random.randn(k,d)
    #b = 2*np.pi*np.random.rand(k,1)
    
    return np.hstack((w,b)).T

def f1(x,w):
    '''
    This function computes random fourier features
    param:
    - x = the data, horizontally stacked
    - w = w and b random parameters to compute features, given as a tuple (x,b)
    '''
    return np.cos(np.dot(np.hstack((x,np.ones((x.shape[0],1)))),w)).mean(0)

def generate_random_features_all(t,y,num_feat):
    return np.hstack((generate_random_features_1d(t,num_feat),
                     generate_random_features_1d(y,num_feat),
                     generate_random_features_2d(t,y,num_feat)))

def generate_random_features(t, y, num_feat):
    '''
    This functions generates random features for a set of paired inputs, such as times and observations
    Inputs:
    - t: first dimension of data, (n x d) array
    - y: second dimension of data, (n x d) array
    - num_feat: number of random features to be generated
    Output:
    - features in fixed dimensional space (n x num_feat) array
    '''
    # if different sizes do interpolation and coerce to equal number of dimensions
    if isinstance(t,list):
        # find length-scale for random features using median heuristic
        mean_obs = int(np.mean([len(row) for row in t]))
        np.random.seed(0)
        t_sig = np.array([x[np.sort(np.random.choice(range(len(x)), mean_obs, replace=False))] 
                          for x in t if len(x) > mean_obs]) 
        y_sig = np.array([x[np.sort(np.random.choice(range(len(x)), mean_obs, replace=False))]  
                          for x in y if len(x) > mean_obs]) 
        
        sig = meddistance(np.hstack((t_sig, y_sig)), subsample=1000)
        np.random.seed()
    
    else:
        sig = meddistance(np.hstack((t, y)), subsample=1000)

    return np.array([f1(np.hstack((t[:,np.newaxis],y[:,np.newaxis])),rp(num_feat,sig,2)) for (t,y) in zip(t,y)])

def generate_random_features_2d(t, y, num_feat, sig = None):
    '''
    This functions generates random features for a set of paired inputs, such as times and observations
    Inputs:
    - t: first dimension of data, (n x d) array
    - y: second dimension of data, (n x d) array
    - num_feat: number of random features to be generated
    Output:
    - features in fixed dimensional space (n x num_feat) array
    '''
    if sig == None:
        # find length-scale for random features using median heuristic
        sig = meddistance(np.hstack((t, y)), subsample=1000)

    return np.array([f1(np.hstack((t[:,np.newaxis],y[:,np.newaxis])),rp(num_feat,sig,2)) for (t,y) in zip(t,y)])

def generate_random_features_2d_ind(t, y, num_feat, sig = None):
    '''
    This functions generates random features for a set of paired inputs, such as times and observations
    Inputs:
    - t: first dimension of data, (n x d) array
    - y: second dimension of data, (n x d) array
    - num_feat: number of random features to be generated
    Output:
    - features in fixed dimensional space (n x num_feat) array
    '''
    if sig == None:
        # find length-scale for random features using median heuristic
        sig = meddistance(np.hstack((t, y)), subsample=1000)
    random_parameters = rp_ind(num_feat,sig,2)
    return np.array([f1(np.hstack((t[:,np.newaxis],y[:,np.newaxis])),random_parameters) for (t,y) in zip(t,y)])

def generate_random_features_1d(data, num_feat):
    '''
    This functions generates random features for a set of paired inputs, such as times and observations
    Inputs:
    - data:  (n x d) array
    - num_feat: number of random features to be generated
    Output:
    - features in fixed dimensional space (n x num_feat) array
    '''
    # find length-scale for random features using median heuristic
    if isinstance(data,list):
        mean_obs = int(np.mean([len(row) for row in data]))
        data_sig = np.array([x[np.sort(np.random.choice(range(len(x)), mean_obs, replace=False))] 
                          for x in data if len(x) >= mean_obs])
        
        sig = meddistance(np.concatenate(data_sig)[:,np.newaxis], subsample=1000)
    else:
        sig = meddistance(data, subsample=1000)
            
    return np.array([f1(row[:,np.newaxis],rp(num_feat,sig,1)) for row in data])

def generate_random_features_1d_ind(data, num_feat):
    '''
    This functions generates random features for a set of paired inputs, such as times and observations
    Inputs:
    - data:  (n x d) array
    - num_feat: number of random features to be generated
    Output:
    - features in fixed dimensional space (n x num_feat) array
    '''
    # find length-scale for random features using median heuristic
    sig = meddistance(data, subsample=1000)
    random_parameters = rp_ind(num_feat,sig,1)
    
    return np.array([f1(row[:,np.newaxis],random_parameters) for row in data])

def generate_random_features_ind(data1, data2, num_feat):
    '''
    This functions generates random features for a set of paired inputs, such as times and observations
    Inputs:
    - data:  (n x d) array
    - num_feat: number of random features to be generated
    Output:
    - features in fixed dimensional space (n x num_feat) array
    '''
    # find length-scale for random features using median heuristic
    sig = meddistance(np.vstack((data1,data2)), subsample=1000)
    random_parameters = rp_ind(num_feat,sig,1)
    rff1 = np.array([f1(row[:,np.newaxis],random_parameters) for row in data1])
    rff2 = np.array([f1(row[:,np.newaxis],random_parameters) for row in data2])
    return rff1, rff2