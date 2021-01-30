"""
This script contains many different independence tests' 

We acknowledge the use of existing code from the repositories in https://github.com/wittawatj/ that served as a skeleton for the implementation of some of our tests, especially kernel-based tests. Thank you!
"""

from __future__ import print_function
from __future__ import division

from builtins import str
from builtins import range
from past.utils import old_div
from builtins import object

import general_utils as general_utils
import kernel_utils as kernel_utils
import data as data

import autograd
import autograd.numpy as np
import tensorflow as tf
import scipy
import scipy.stats as stats
import math
from random import sample
from scipy.stats import norm as normal
from tensorflow.python.framework import ops
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic, ExpSineSquared, WhiteKernel)
from sklearn.gaussian_process.kernels import ConstantKernel as C

from numpy import eye, sqrt, trace, diag, zeros
from scipy.stats import chi2, gamma
from kernel_utils import centering, pdinv, truncated_eigen, eigdec, columnwise_normalizes, rbf_kernel_median
from scipy.interpolate import interp1d, splrep,splev
from scipy.stats import rankdata


class QuadHSICTest(object):
    """
    Quadratic MMD test where the null distribution is computed by permutation.
    - Use a single U-statistic i.e., remove diagonal from the Kxy matrix.
    - The code is based on a Matlab code of Arthur Gretton from the paper 
    A TEST OF RELATIVE SIMILARITY FOR MODEL SELECTION IN GENERATIVE MODELS
    ICLR 2016
    """

    def __init__(self, k, l):
        """
        kernel: an instance of Kernel 
        n_permute: number of times to do permutation
        """
        self.k = k
        self.l = l
        
    def perform_test(self, tst_data, alpha):
        """perform the two-sample test and return values computed in a dictionary:
        {alpha: 0.01, pvalue: 0.0002, test_stat: 2.3, h0_rejected: True, ...}
        tst_data: an instance of TSTData
        """
        hsic_stat = self.compute_stat(tst_data)
        
        # approximate p-value with bootstrap permutations 
        #pvalue = self.hsic_null_gamma(tst_data, alpha)
        X, Y = tst_data.xy()
        k = self.k
        l = self.l
        list_hsic = QuadHSICTest.list_permute(X, Y, k, l)
        pvalue = np.mean(list_hsic > hsic_stat)
        
        results = {'alpha': alpha, 'pvalue': pvalue, 'test_stat': hsic_stat,
                'h0_rejected': pvalue < alpha, 'permutation_list': list_hsic}
        return results
    
    
    def compute_pvalue(self, tst_data, alpha):
        """perform the independence test and return p value
        tst_data: an instance of TSTData
        alpha: significance threshold
        """
        hsic_stat = self.compute_stat(tst_data)
        
        # approximate p-value with bootstrap permutations 
        #pvalue = self.hsic_null_gamma(tst_data, alpha)
        X, Y = tst_data.xy()
        k = self.k
        l = self.l
        list_hsic = QuadHSICTest.list_permute(X, Y, k, l)
        pvalue = np.mean(list_hsic > hsic_stat)
        
        return pvalue
    
    def compute_stat(self, tst_data):
        """Compute the test statistic: given as the empirical quadratic n*HSIC
        The HSIC is defined as Tr(KHLH) but all implementations use sum to compute it"""
        X, Y = tst_data.xy()
        nx = X.shape[0]
        ny = Y.shape[0]

        if nx != ny:
            raise ValueError('nx must be the same as ny')

        k = self.k
        l = self.l
        #KX = centering(kx.eval(X,X)) 
        #KY = centering(ky.eval(Y,Y)) 
    
        #hsic = np.sum(KX * KY) / nx
        
        K = k.eval(X, X)
        L = l.eval(Y, Y)
        Kmean = np.mean(K, 0)
        Lmean = np.mean(L, 0)
        HK = K - Kmean
        HL = L - Lmean
        # t = trace(KHLH)
        HKf = HK.flatten()/(nx-1) 
        HLf = HL.T.flatten()/(nx-1)
        hsic = HKf.dot(HLf)
        
        return hsic
        
    
    @staticmethod 
    def permutation_list_hsic(X, Y, k, l, n_permute=400, seed=8273):
        """
        Repeatedly mix, permute X,Y and compute HSIC^2. This is intended to be
        used to approximate the null distritubion.
        """
        #XY = np.vstack((X, Y))

        #sig2 = general_utils.meddistance(XY, subsample=1000)
        #k = kernel_utils.KGauss(sig2)
        
        #Kxyxy = kx.eval(XY, XY)
        K = k.eval(X, X)
        Kmean = np.mean(K, 0)
        HK = K - Kmean

        rand_state = np.random.get_state()
        #np.random.seed(seed)
        
        #nxy = XY.shape[0]
        n = X.shape[0]
        d = X.shape[1]
        list_hsic = np.zeros(n_permute)
        
        for r in range(n_permute):
            ind = np.random.choice(d,d , replace=False)
            L = l.eval(Y[:,ind], Y)
            Lmean = np.mean(L, 0)
        
            HL = L - Lmean
            # t = trace(KHLH)
            HKf = HK.flatten()/(n-1) 
            HLf = HL.T.flatten()/(n-1)
            hsic = HKf.dot(HLf)
            #hsic = np.sum(centering(Kx) * centering(Ky_temp)) / ny
            list_hsic[r] = hsic
            
        np.random.set_state(rand_state)
        return list_hsic
    

    @staticmethod
    def list_permute(X, Y, k, l, n_permute=400, seed=8273):
        """
        Return a numpy array of HSIC's for each permutation.
        This is an implementation where kernel matrices are pre-computed.
        TODO: can be improved.
        """
        if X.shape[0] != Y.shape[0]:
            raise ValueError('X and Y must have the same number of rows (sample size')
        n = X.shape[0]

        r = 0 
        arr_hsic = np.zeros(n_permute)
        K = k.eval(X, X)
        L = l.eval(Y, Y)
        # set the seed 
        rand_state = np.random.get_state()
        np.random.seed(seed)

        while r < n_permute:
            # shuffle the order of X, Y while still keeping the original pairs
            ind = np.random.choice(n, n, replace=False)
            Ks = K[np.ix_(ind, ind)]
            #Xs = X[ind]
            #Ys = Y[ind]
            #Ks2 = k.eval(Xs, Xs)
            #assert np.linalg.norm(Ks - Ks2, 'fro') < 1e-4

            Ls = L[np.ix_(ind, ind)]
            Kmean = np.mean(Ks, 0)
            HK = Ks - Kmean
            HKf = HK.flatten()/(n-1) 
            # shift Ys n-1 times 
            for s in range(n-1):
                if r >= n_permute:
                    break
                Ls = np.roll(Ls, 1, axis=0)
                Ls = np.roll(Ls, 1, axis=1)

                # compute HSIC 
                Lmean = np.mean(Ls, 0)
                HL = Ls - Lmean
                # t = trace(KHLH)
                HLf = HL.T.flatten()/(n-1)
                bhsic = HKf.dot(HLf)

                arr_hsic[r] = bhsic
                r = r + 1
        # reset the seed back 
        np.random.set_state(rand_state)
        return arr_hsic
    
    @staticmethod 
    def _list_permute_generic(X, Y, k, l, n_permute=400, seed=8273):
        """
        Return a numpy array of HSIC's for each permutation.
        This is a naive generic implementation where kernel matrices are 
        not pre-computed.
        """
        if X.shape[0] != Y.shape[0]:
            raise ValueError('X and Y must have the same number of rows (sample size')
        n = X.shape[0]

        r = 0 
        arr_hsic = np.zeros(n_permute)
        # set the seed 
        rand_state = np.random.get_state()
        np.random.seed(seed)
        while r < n_permute:
            # shuffle the order of X, Y while still keeping the original pairs
            ind = np.random.choice(n, n, replace=False)
            Xs = X[ind]
            Ys = Y[ind]
            # shift Ys n-1 times 
            for s in range(n-1):
                if r >= n_permute:
                    break
                Ys = np.roll(Ys, 1, axis=0)
                # compute HSIC 
                bhsic = QuadHSIC.biased_hsic(Xs, Ys, k, l)
                arr_hsic[r] = bhsic
                r = r + 1
        # reset the seed back 
        np.random.set_state(rand_state)
        return arr_hsic
    
    def hsic_null_gamma(self, tst_data, alpha):
        """Approximate the null distribution with a two-parameter Gamma.
        This code computes the alpha and beta parameters of this approximation and computes pvalue"""
        X, Y = tst_data.xy()
        n = X.shape[0]
        m = Y.shape[0]

        if n != m:
            raise ValueError('nx must be the same as ny')

        k = self.k
        l = self.l
        Kx = centering(kx.eval(X,X)) 
        Ky = centering(ky.eval(Y,Y)) 

        testStat = np.sum(centering(Kx) * centering(Ky)) / n

        varHSIC = (Kx * Ky / 6)**2

        varHSIC = ( np.sum(varHSIC) - np.trace(varHSIC) ) / n / (n-1)

        varHSIC = varHSIC * 72 * (n-4) * (n-5) / n / (n-1) / (n-2) / (n-3)

        Kx = Kx - np.diag(np.diag(Kx))
        Ky = Ky - np.diag(np.diag(Ky))

        bone = np.ones((n, 1), dtype = float)
        muX = np.dot(np.dot(bone.T, Kx), bone) / n / (n-1)
        muY = np.dot(np.dot(bone.T, Ky), bone) / n / (n-1)

        mHSIC = (1 + muX * muY - muX - muY) / n

        al = mHSIC**2 / varHSIC # alpha
        bet = varHSIC*n / mHSIC # beta

        thresh = gamma.ppf(1-alpha, al, scale=bet)[0][0] # threshold for significance
        pvalue = 1 - gamma.cdf(testStat, al, scale=bet)[0][0] # pvalue of test

        return pvalue

    @staticmethod
    def h1_mean_var(X, Y, k, l):
        """
        X: nxd numpy array 
        Y: nxd numpy array
        k, l: a Kernel object 
        
        return (HSIC, var[HSIC]) under H1
        """
        m = X.shape[0]
        n = Y.shape[0]

        if m != n:
            raise ValueError('length X should be the same as length Y')

        K = k.eval(X,X) 
        L = l.eval(Y,Y)
        Kmean = np.mean(K, 0)
        Lmean = np.mean(L, 0)
        
        HK = K - Kmean
        HL = L - Lmean
        # t = trace(KHLH)
        HKf = HK.flatten()/(n-1) 
        HLf = HL.T.flatten()/(n-1)
        hsic = HKf.dot(HLf)
        #hsic = np.sum(centering(KX) * centering(KY)) / m**2
        
        # variance computation
        var = 0
        for i in range(m):
            var1 = np.inner(K[i,:],L[i,:]) # inner product between vectors
            var2 = (L.sum() * K[i,:]).sum()
            var3 = np.outer(K[i,:],L[i,:]).sum()
            var += ((var1 / m + var2 / (m**3) - 2 * var3 / (m**2)) **2 ) / m
            
        hsic_var = max(16 * (var - hsic**2),1e-3) 
        
        # variance computation 2
        #KX = KX - np.diag(KX); KY = KY - np.diag(KY)
        #KXKY = KX @ KY; KYKX = KY @ KX
        #ones = np.ones(m)
        #h1 = (m-2)**2 * (np.multiply(KX,KY) @ ones)
        #h2 = (m-2) * (np.trace(KXKY) - KXKY @ ones - KYKX @ ones )
        #print(h2)
        #h3 = m * np.multiply(KX @ ones, KY @ ones)
        #print(h3)
        #h4 = (ones @ KY @ ones) * (KX @ ones)
        #print(h4)
        #h5 = (ones @ KX @ ones) * (KY @ ones)
        #print(h5)
        #h6 = (ones @ KXKY @ ones) 
        #h = h1 + h2 -h3 + h4 + h5 - h6
        #print(h @ h / (4*m*((m-1)*(m-2)*(m-3))**2))
        
        return hsic, hsic_var

    @staticmethod
    def grid_search_kernel(tst_data, list_kernels_x, list_kernels_y, reg=1e-5):
        """
        Return from the list the best kernels for X and Y that maximizes the test power criterion.
        
        In principle, the test threshold depends on the null distribution, which 
        changes with kernel. Thus, we need to recompute the threshold for each kernel
        (require permutations), which is expensive. However, asymptotically 
        the threshold goes to 0. So, for each kernel, the criterion needed is
        the ratio mean/variance of the HSIC. (Source: Arthur Gretton)
        This is an approximate to avoid doing permutations for each kernel 
        candidate.
        - reg: regularization parameter
        return: (best kernel index, list of test power objective values)
        """
        import time
        X, Y = tst_data.xy()
        n = X.shape[0]
        n = len(list_kernels_x)
        m = len(list_kernels_y)
        if m != n:
            raise ValueError('list kernels X should be the same length as list kernels Y')
            
        obj_values_x = obj_values_y = np.zeros(n**2)
        count = 0
        for k in list_kernels_x:
            for l in list_kernels_y:
                start = time.time()
                hsic, hsic_var = QuadHSICTest.h1_mean_var(X, Y, k, l)
                obj = float(hsic)/((hsic_var + reg)**0.5)
                obj_values_x[count] = obj
                obj_values_y[count] = obj
                end = time.time()
                #print('(%d/%d) k %s: l %s: hsic: %.3g, var: %.3g, power obj: %g, took: %s'%(count+1,
                #    n**2, str(k), str(l), hsic, hsic_var, obj, end-start))
                count += 1
        best_ind_x = int(np.argmax(obj_values_x) / n)
        best_ind_y = np.argmax(obj_values_y) % n
        return best_ind_x, best_ind_y, obj_values_x, obj_values_y
    


    
def RHSIC(t1,y1,t2,y2,alpha=0.01,output = 'stat',opt_kernel=True):
    '''
    Runs full test with all optimization procedures included
    output: the desired output, one of 'stat', 'p_value', 'full'
    '''
    # rescale data
    max_y = max(np.concatenate((abs(y1.flatten()),abs(y2.flatten()))))
    y1 = y1/max(abs(y1.flatten()))
    y2 = y2/max(abs(y2.flatten()))
    
    sig = general_utils.meddistance(np.vstack((np.hstack((t1, y1)),np.hstack((t2, y2)))), subsample=1000)
    
    # generate random features
    X, Y = general_utils.generate_random_features_ind(y1,y2,num_feat= 20)
    
    w1, w2 = [len(t) for t in y1], [len(t) for t in y2]
    tst_data = data.TSTData(X, Y, w1, w2)
    tr, te = tst_data.split_tr_te(tr_proportion=0.5)
    xtr, ytr = tr.xy()
    
    # Compute median pairwise distance
    med_x = general_utils.meddistance(xtr, 1000)
    med_y = general_utils.meddistance(ytr, 1000)
    
    if opt_kernel == False:
        best_ker_x = kernel_utils.KGauss(med_x**2)
        best_ker_y = kernel_utils.KGauss(med_y**2)
    else:    
        list_gwidth = np.hstack( ( (med_x**2) *(2.0**np.linspace(-1, 1, 5) ) ) )
        list_gwidth.sort()
        list_kernels_x = [kernel_utils.KGauss(gw2) for gw2 in list_gwidth]

        list_gwidth = np.hstack( ( (med_y**2) *(2.0**np.linspace(-1, 1, 5) ) ) )
        list_gwidth.sort()
        list_kernels_y = [kernel_utils.KGauss(gw2) for gw2 in list_gwidth]

        # grid search to choose the best Gaussian width
        bestix, bestiy, _, _ = QuadHSICTest.grid_search_kernel(tr, list_kernels_x, list_kernels_y)

        best_ker_x = list_kernels_x[bestix]
        best_ker_y = list_kernels_y[bestiy]
        
    hsic_test = QuadHSICTest(best_ker_x, best_ker_y)

    if output == 'stat':
        return hsic_test.compute_stat(te)
    if output == 'p_value':
        return hsic_test.compute_pvalue(te,alpha=alpha)
    if output == 'full':
        return hsic_test.perform_test(te,alpha=alpha)
    
def HSIC(t1,y1,t2,y2,alpha=0.01,output = 'stat',opt_kernel=True):
    '''
    Runs full test with all optimization procedures included
    output: the desired output, one of 'stat', 'p_value', 'full'
    '''
    
    # interpolate
    t1, X = interpolate(t1,y1)
    t2, Y = interpolate(t2,y2)
    
    w1, w2 = [len(t) for t in y1], [len(t) for t in y2]
    tst_data = data.TSTData(X, Y, w1, w2)
    tr, te = tst_data.split_tr_te(tr_proportion=0.5)
    xtr, ytr = tr.xy()
    
    # Compute median pairwise distance
    med_x = general_utils.meddistance(xtr, 1000)
    med_y = general_utils.meddistance(ytr, 1000)
     
    if opt_kernel == False:
        best_ker_x = kernel_utils.KGauss(med_x**2)
        best_ker_y = kernel_utils.KGauss(med_y**2)
    else:    
        list_gwidth = np.hstack( ( (med_x**2) *(2.0**np.linspace(-1, 1, 5) ) ) )
        list_gwidth.sort()
        list_kernels_x = [kernel_utils.KGauss(gw2) for gw2 in list_gwidth]

        list_gwidth = np.hstack( ( (med_y**2) *(2.0**np.linspace(-1, 1, 5) ) ) )
        list_gwidth.sort()
        list_kernels_y = [kernel_utils.KGauss(gw2) for gw2 in list_gwidth]

        # grid search to choose the best Gaussian width
        bestix, bestiy, _, _ = QuadHSICTest.grid_search_kernel(tr, list_kernels_x, list_kernels_y)

        best_ker_x = list_kernels_x[bestix]
        best_ker_y = list_kernels_y[bestiy]
        
    hsic_test = QuadHSICTest(best_ker_x, best_ker_y)

    if output == 'stat':
        return hsic_test.compute_stat(te)
    if output == 'p_value':
        return hsic_test.compute_pvalue(te,alpha=alpha)
    if output == 'full':
        return hsic_test.perform_test(te,alpha=alpha)
    
def interpolate(t,y,num_obs=50):
    """
    Interpolates each trajectory such that observation times coincide for each one.
    
    Note: initially cubic interpolation gave great power, but this happens as an artifact of the interpolation,
    as both trajectories have the same number of observations. Type I error was increased as a result. To avoid 
    this we settled for a linear interpolation between observations.
    Splines were also tried but gave very bad interpolations.
    """
    
    t = np.array([np.sort(row) for row in t])
    t = np.insert(t, 0, 0, axis=1)
    t = np.insert(t, len(t[0]), 1, axis=1)
    y = np.insert(y, 0, y[:,0], axis=1)
    y = np.insert(y, len(y[0]), y[:,-1], axis=1)
    
    new_t = np.zeros(num_obs)
    new_y = np.zeros(num_obs)
    
    for i in range(len(t)):
        f = interp1d(t[i], y[i], kind='linear')
        #f = splrep(t[i], y[i])
        t_temp = np.random.uniform(low=0.0, high=1.0, size=num_obs)#np.linspace(0.1,0.9,num_obs)
        y_temp = f(t_temp)
        #y_temp = splev(t_temp, f, der=0)
        new_y = np.vstack((new_y, y_temp))
        new_t = np.vstack((new_t, t_temp))
        
    return new_t[1:], new_y[1:]
    
def rdc(x, y, f=np.sin, k=20, s=1/6., n=1):
    """
    Computes the Randomized Dependence Coefficient
    x,y: numpy arrays 1-D or 2-D
         If 1-D, size (samples,)
         If 2-D, size (samples, variables)
    f:   function to use for random projection
    k:   number of random projections to use
    s:   scale parameter
    n:   number of times to compute the RDC and
         return the median (for stability)
    According to the paper, the coefficient should be relatively insensitive to
    the settings of the f, k, and s parameters.
    
    Source: https://github.com/garydoranjr/rdc
    """
    #x = x.reshape((len(x)))
    #y = y.reshape((len(y)))
    
    if n > 1:
        values = []
        for i in range(n):
            try:
                values.append(rdc(x, y, f, k, s, 1))
            except np.linalg.linalg.LinAlgError: pass
        return np.median(values)

    if len(x.shape) == 1: x = x.reshape((-1, 1))
    if len(y.shape) == 1: y = y.reshape((-1, 1))

    # Copula Transformation
    cx = np.column_stack([rankdata(xc, method='ordinal') for xc in x.T])/float(x.size)
    cy = np.column_stack([rankdata(yc, method='ordinal') for yc in y.T])/float(y.size)

    # Add a vector of ones so that w.x + b is just a dot product
    O = np.ones(cx.shape[0])
    X = np.column_stack([cx, O])
    Y = np.column_stack([cy, O])

    # Random linear projections
    Rx = (s/X.shape[1])*np.random.randn(X.shape[1], k)
    Ry = (s/Y.shape[1])*np.random.randn(Y.shape[1], k)
    X = np.dot(X, Rx)
    Y = np.dot(Y, Ry)

    # Apply non-linear function to random projections
    fX = f(X)
    fY = f(Y)

    # Compute full covariance matrix
    C = np.cov(np.hstack([fX, fY]).T)

    # Due to numerical issues, if k is too large,
    # then rank(fX) < k or rank(fY) < k, so we need
    # to find the largest k such that the eigenvalues
    # (canonical correlations) are real-valued
    k0 = k
    lb = 1
    ub = k
    while True:

        # Compute canonical correlations
        Cxx = C[:k, :k]
        Cyy = C[k0:k0+k, k0:k0+k]
        Cxy = C[:k, k0:k0+k]
        Cyx = C[k0:k0+k, :k]

        eigs = np.linalg.eigvals(np.dot(np.dot(np.linalg.pinv(Cxx), Cxy),
                                        np.dot(np.linalg.pinv(Cyy), Cyx)))

        # Binary search if k is too large
        if not (np.all(np.isreal(eigs)) and
                0 <= np.min(eigs) and
                np.max(eigs) <= 1):
            ub -= 1
            k = (ub + lb) // 2
            continue
        if lb == ub: break
        lb = k
        if ub == lb + 1:
            k = ub
        else:
            k = (ub + lb) // 2

    return np.sqrt(np.max(eigs))

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------


def RDC(t1,y1,t2,y2,alpha=0.01,output = 'stat'):
    t1, y1 = interpolate(t1,y1)
    t2, y2 = interpolate(t2,y2)
    
    stat = rdc(y1.flatten(),y2.flatten())
    
    # permutations
    n_permute = 200
    n = len(y1)
    list_stat = np.zeros(n_permute)
    
    for r in range(n_permute):
        ind = np.random.choice(n, n, replace=False)
        y1_new = y1[ind]
        
        stat_null = rdc(y1_new.flatten(),y2.flatten())
        list_stat[r] = stat_null
    
    #print(list_stat)
    if output == 'stat':
        return stat
    if output == 'p_value':
        return np.mean(list_stat > stat)
    if output == 'full':
        return {'alpha': alpha, 'pvalue': np.mean(list_stat > stat), 'test_stat': stat,
                'h0_rejected': np.mean(list_stat > stat) < alpha}
    
    
    
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

def COR(t1,y1,t2,y2,alpha=0.01,output = 'stat'):
    t1, y1 = interpolate(t1,y1)
    t2, y2 = interpolate(t2,y2)
    stat = np.abs(np.corrcoef(y1.flatten(),y2.flatten())[0, 1])
    
    # permutations
    n_permute = 200
    n = len(y1)
    list_stat = np.zeros(n_permute)
    y1y2 = np.vstack((y1,y2))
    
    for r in range(n_permute):
        ind = np.random.choice(n, n, replace=False)
        y1_new = y1[ind]
        
        stat_null = np.abs(np.corrcoef(y1_new.flatten(),y2.flatten())[0, 1])
        list_stat[r] = stat_null
    
    #print(list_stat)
    if output == 'stat':
        return stat
    if output == 'p_value':
        return np.mean(list_stat > stat)
    if output == 'full':
        return {'alpha': alpha, 'pvalue': np.mean(list_stat > stat), 'test_stat': stat,
                'h0_rejected': np.mean(list_stat > stat) < alpha}


# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

def dCov(t1,y1,t2,y2,alpha=0.01,output = 'stat'):
    ''' 
    Compute permutation test using the distance correlation.
    source : https://gist.github.com/josef-pkt/2938402
    '''
    def dist(x, y):
        #1d only
        return np.abs(x[:, None] - y)
    

    def d_n(x):
        d = dist(x, x)
        dn = d - d.mean(0) - d.mean(1)[:,None] + d.mean()
        return dn


    def dcov(x, y):
        dnx = d_n(x)
        dny = d_n(y)

        denom = np.product(dnx.shape)
        dc = (dnx * dny).sum() / denom
        dvx = (dnx**2).sum() / denom
        dvy = (dny**2).sum() / denom
        dr = dc / (np.sqrt(dvx) * np.sqrt(dvy))
        return dc #, dr, dvx, dvy

    t1, y1 = interpolate(t1,y1)
    t2, y2 = interpolate(t2,y2)
    stat = dcov(y1.flatten(),y2.flatten())
    
    # permutations
    n_permute = 200
    n = len(y1)
    list_stat = np.zeros(n_permute)
    
    for r in range(n_permute):
        ind = np.random.choice(n, n, replace=False)
        y1_new = y1[ind]
        
        stat_null = dcov(y1_new.flatten(),y2.flatten())
        list_stat[r] = stat_null
    
    #print(list_stat)
    if output == 'stat':
        return stat
    if output == 'p_value':
        return np.mean(list_stat > stat)
    if output == 'full':
        return {'alpha': alpha, 'pvalue': np.mean(list_stat > stat), 'test_stat': stat,
                'h0_rejected': np.mean(list_stat > stat) < alpha}

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------



    
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
# This is an extension to conditional independence (in progress)
    
def python_kcit(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, alpha=0.05, noise=1e-3, num_bootstrap_for_null=5000, normalize=True, kern=rbf_kernel_median, seed=None):
    if seed is not None:
        np.random.seed(seed)

    T = len(Y)

    if normalize:
        X, Y, Z = columnwise_normalizes(X, Y, Z)

    Kx = centering(kern(np.hstack([X, Z])))  # originally [x, z /2]
    Ky = centering(kern(Y))

    Kz = centering(kern(Z))
    P1 = eye(T) - Kz @ pdinv(Kz + noise * eye(T))  # pdinv(I+K/noise)
    Kxz = P1 @ Kx @ P1.T
    Kyz = P1 @ Ky @ P1.T

    test_statistic = (Kxz * Kyz).sum()  # trace(Kxz @ Kyz)

    # null computation
    return kcit_null(Kxz, Kyz, T, alpha, num_bootstrap_for_null, test_statistic)


def python_kcit_K(Kx: np.ndarray, Ky: np.ndarray, Kz: np.ndarray, alpha=0.05, sigma_squared=1e-3, num_bootstrap_for_null=5000, seed=None):
    if seed is not None:
        np.random.seed(seed)

    T = len(Kx)

    Kx, Ky, Kz = centering(Kx * Kz), centering(Ky), centering(Kz)

    P = eye(T) - Kz @ pdinv(Kz + sigma_squared * eye(T))
    Kxz = P @ Kx @ P.T
    Kyz = P @ Ky @ P.T

    test_statistic = (Kxz * Kyz).sum()  # trace(Kxz @ Kyz)

    return kcit_null(Kxz, Kyz, T, alpha, num_bootstrap_for_null, test_statistic)


def python_kcit_K2(Kx: np.ndarray, Ky: np.ndarray, Z: np.ndarray, alpha=0.05, with_gp=False, sigma_squared=1e-3, num_bootstrap_for_null=5000, seed=None):
    if seed is not None:
        np.random.seed(seed)

    T = len(Kx)

    Kz = rbf_kernel_median(Z)
    Kx, Ky, Kz = centering(Kx * Kz), centering(Ky), centering(Kz)

    P = eye(T) - Kz @ pdinv(Kz + sigma_squared * eye(T))
    Kxz = P @ Kx @ P.T
    Kyz = P @ Ky @ P.T

    test_statistic = (Kxz * Kyz).sum()  # trace(Kxz @ Kyz)

    return kcit_null(Kxz, Kyz, T, alpha, num_bootstrap_for_null, test_statistic)


def kcit_null(Kxz, Kyz, T, alpha, num_bootstrap_for_null, test_statistic):
    # null computation
    eig_Kxz, eivx = truncated_eigen(*eigdec(Kxz))
    eig_Kyz, eivy = truncated_eigen(*eigdec(Kyz))

    eiv_prodx = eivx @ diag(sqrt(eig_Kxz))
    eiv_prody = eivy @ diag(sqrt(eig_Kyz))

    num_eigx = eiv_prodx.shape[1]
    num_eigy = eiv_prody.shape[1]
    size_u = num_eigx * num_eigy
    uu = zeros((T, size_u))
    for i in range(num_eigx):
        for j in range(num_eigy):
            uu[:, i * num_eigy + j] = eiv_prodx[:, i] * eiv_prody[:, j]

    uu_prod = uu @ uu.T if size_u > T else uu.T @ uu
    eig_uu = truncated_eigen(eigdec(uu_prod, min(T, size_u))[0])

    boot_critical_val, boot_p_val = _null_by_bootstrap(test_statistic, num_bootstrap_for_null, alpha, eig_uu[:, None])
    appr_critical_val, appr_p_val = _null_by_gamma_approx(test_statistic, alpha, uu_prod)

    return test_statistic, boot_critical_val, boot_p_val, appr_critical_val, appr_p_val




