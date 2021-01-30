"""
This script contains various two-sample tests used in the experiments of 'Hypothesis Testing with Uncertain Curves' 

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
tf.compat.v1.disable_eager_execution()
import scipy
import scipy.stats as stats
import math
from random import sample
from scipy.stats import norm as normal
from tensorflow.python.framework import ops
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic, ExpSineSquared, WhiteKernel)
from sklearn.gaussian_process.kernels import ConstantKernel as C
from scipy.interpolate import interp1d
from scipy.stats import rankdata
from numpy.linalg import inv


class HotellingT2Test(object):
    """Two-sample test with Hotelling T-squared statistic.
    Technical details follow "Applied Multivariate Analysis" of Neil H. Timm.
    See page 156.

    """

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def perform_test(self, tst_data):
        """Perform the two-sample test and return values computed in a dictionary:
        {alpha: 0.01, pvalue: 0.0002, test_stat: 2.3, h0_rejected: True, ...}
        tst_data: an instance of TSTData
        """
        d = tst_data.dim()
        chi2_stat = self.compute_stat(tst_data)
        pvalue = stats.chi2.sf(chi2_stat, d)
        alpha = self.alpha
        results = {'alpha': self.alpha, 'pvalue': pvalue, 'test_stat': chi2_stat,
                   'h0_rejected': pvalue < alpha}
        return results
    
    def compute_pvalue(self,tst_data):
        d = tst_data.dim()
        chi2_stat = self.compute_stat(tst_data)
        return stats.chi2.sf(chi2_stat, d)
        
    def compute_stat(self, tst_data):
        """Compute the test statistic"""
        X, Y = tst_data.xy()
        if X.shape[0] != Y.shape[0]:
            raise ValueError('Require nx = ny for now. Will improve if needed.')
        nx = X.shape[0]
        ny = Y.shape[0]
        mx = np.mean(X, 0)
        my = np.mean(Y, 0)
        mdiff = mx - my
        sx = np.cov(X.T)
        sy = np.cov(Y.T)
        s = old_div(sx, nx) + old_div(sy, ny)
        chi2_stat = np.dot(np.linalg.solve(s, mdiff), mdiff)
        return chi2_stat

def T2(t1,y1,t2,y2,alpha=0.01,output = 'stat'):
    '''
    Runs full test with all optimization procedures included
    output: the desired output, one of 'stat', 'p_value', 'full'
    '''
    
    # interpolate
    t1, x1 = interpolate(t1,y1)
    t2, x2 = interpolate(t2,y2)
    
    t2_test = HotellingT2Test(alpha=alpha)
    if output == 'stat':
        return t2_test.compute_stat(te)
    if output == 'p_value':
        return t2_test.compute_pvalue(te)
    if output == 'full':
        return t2_test.perform_test(te)
    
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------


class LinearMMDTest(object):
    """Two-sample test with linear MMD^2 statistic.
    """

    def __init__(self, kernel, alpha=0.01):
        """
        kernel: an instance of Kernel
        """
        self.kernel = kernel
        self.alpha = alpha

    def perform_test(self, tst_data):
        """perform the two-sample test and return values computed in a dictionary:
        {alpha: 0.01, pvalue: 0.0002, test_stat: 2.3, h0_rejected: True, ...}
        tst_data: an instance of TSTData
        """
        X, Y = tst_data.xy()
        n = X.shape[0]
        stat, snd = LinearMMDTest.two_moments(X, Y, self.kernel)
        # var = snd - stat**2
        var = snd
        pval = stats.norm.sf(stat, loc=0, scale=(2.0 * var / n) ** 0.5)
        results = {'alpha': self.alpha, 'pvalue': pval, 'test_stat': stat,
                   'h0_rejected': pval < self.alpha}
        return results
    
    def compute_pvalue(self, tst_data):
        """perform the two-sample test and return p-values 
        tst_data: an instance of TSTData
        """
        X, Y = tst_data.xy()
        n = X.shape[0]
        stat, snd = LinearMMDTest.two_moments(X, Y, self.kernel)
        # var = snd - stat**2
        var = snd
        pval = stats.norm.sf(stat, loc=0, scale=(2.0 * var / n) ** 0.5)
        
        return pval

    def compute_stat(self, tst_data):
        """Compute unbiased linear mmd estimator."""
        X, Y = tst_data.xy()
        return LinearMMDTest.linear_mmd(X, Y, self.kernel)

    @staticmethod
    def linear_mmd(X, Y, kernel):
        """Compute linear mmd estimator. O(n)"""
        lin_mmd, _ = LinearMMDTest.two_moments(X, Y, kernel)
        return lin_mmd

    @staticmethod
    def two_moments(X, Y, kernel):
        """Compute linear mmd estimator and a linear estimate of
        the uncentred 2nd moment of h(z, z'). Total cost: O(n).
        return: (linear mmd, linear 2nd moment)
        """
        if X.shape[0] != Y.shape[0]:
            raise ValueError('Require sample size of X = size of Y')
        n = X.shape[0]
        if n % 2 == 1:
            # make it even by removing the last row
            X = np.delete(X, -1, axis=0)
            Y = np.delete(Y, -1, axis=0)

        Xodd = X[::2, :]
        Xeven = X[1::2, :]
        assert Xodd.shape[0] == Xeven.shape[0]
        Yodd = Y[::2, :]
        Yeven = Y[1::2, :]
        assert Yodd.shape[0] == Yeven.shape[0]
        # linear mmd. O(n)
        xx = kernel.pair_eval(Xodd, Xeven)
        yy = kernel.pair_eval(Yodd, Yeven)
        xo_ye = kernel.pair_eval(Xodd, Yeven)
        xe_yo = kernel.pair_eval(Xeven, Yodd)
        h = xx + yy - xo_ye - xe_yo
        lin_mmd = np.mean(h)
        """
        Compute a linear-time estimate of the 2nd moment of h = E_z,z' h(z, z')^2.
        Note that MMD = E_z,z' h(z, z').
        Require O(n). Same trick as used in linear MMD to get O(n).
        """
        lin_2nd = np.mean(h ** 2)
        return lin_mmd, lin_2nd

    @staticmethod
    def variance(X, Y, kernel, lin_mmd=None):
        """Compute an estimate of the variance of the linear MMD.
        Require O(n^2). This is the variance under H1.
        """
        if X.shape[0] != Y.shape[0]:
            raise ValueError('Require sample size of X = size of Y')
        n = X.shape[0]
        if lin_mmd is None:
            lin_mmd = LinearMMDTest.linear_mmd(X, Y, kernel)
        # compute uncentred 2nd moment of h(z, z')
        K = kernel.eval(X, X)
        L = kernel.eval(Y, Y)
        KL = kernel.eval(X, Y)
        snd_moment = old_div(np.sum((K + L - KL - KL.T) ** 2), (n * (n - 1)))
        var_mmd = 2.0 * (snd_moment - lin_mmd ** 2)
        return var_mmd

    @staticmethod
    def grid_search_kernel(tst_data, list_kernels, alpha):
        """
        Return from the list the best kernel that maximizes the test power.
        return: (best kernel index, list of test powers)
        """
        X, Y = tst_data.xy()
        n = X.shape[0]
        powers = np.zeros(len(list_kernels))
        for ki, kernel in enumerate(list_kernels):
            lin_mmd, snd_moment = LinearMMDTest.two_moments(X, Y, kernel)
            var_lin_mmd = (snd_moment - lin_mmd ** 2)
            # test threshold from N(0, var)
            thresh = stats.norm.isf(alpha, loc=0, scale=(2.0 * var_lin_mmd / n) ** 0.5)
            power = stats.norm.sf(thresh, loc=lin_mmd, scale=(2.0 * var_lin_mmd / n) ** 0.5)
            # power = lin_mmd/var_lin_mmd
            powers[ki] = power
        best_ind = np.argmax(powers)
        return best_ind, powers


    
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------



class QuadMMDTest(object):
    """
    Quadratic MMD test where the null distribution is computed by permutation.
    - Use a single U-statistic i.e., remove diagonal from the Kxy matrix.
    - The code is based on a Matlab code of Arthur Gretton from the paper 
    A TEST OF RELATIVE SIMILARITY FOR MODEL SELECTION IN GENERATIVE MODELS
    ICLR 2016
    """

    def __init__(self, kernel, n_permute=400, alpha=0.01, use_1sample_U=False):
        """
        kernel: an instance of Kernel 
        n_permute: number of times to do permutation
        """
        self.kernel = kernel
        self.n_permute = n_permute
        self.alpha = alpha 
        self.use_1sample_U = use_1sample_U

    def perform_test(self, tst_data):
        """perform the two-sample test and return values computed in a dictionary:
        {alpha: 0.01, pvalue: 0.0002, test_stat: 2.3, h0_rejected: True, ...}
        tst_data: an instance of TSTData
        """
        d = tst_data.dim()
        alpha = self.alpha
        mmd2_stat = self.compute_stat(tst_data, use_1sample_U=self.use_1sample_U)

        X, Y = tst_data.xy()
        wx, wy = tst_data.weights()
        k = self.kernel
        repeats = self.n_permute
        list_mmd2 = QuadMMDTest.permutation_list_mmd2(X, Y, wx, wy, k, repeats)
        # approximate p-value with the permutations 
        pvalue = np.mean(list_mmd2 > mmd2_stat)

        results = {'alpha': self.alpha, 'pvalue': pvalue, 'test_stat': mmd2_stat,
                'h0_rejected': pvalue < alpha}#, 'list_permuted_mmd2': list_mmd2}
        return results
    
    def compute_pvalue(self, tst_data):
        """perform the two-sample test and return values computed in a dictionary:
        {alpha: 0.01, pvalue: 0.0002, test_stat: 2.3, h0_rejected: True, ...}
        tst_data: an instance of TSTData
        """
        d = tst_data.dim()
        alpha = self.alpha
        mmd2_stat = self.compute_stat(tst_data, use_1sample_U=self.use_1sample_U)

        X, Y = tst_data.xy()
        wx, wy = tst_data.weights()
        k = self.kernel
        repeats = self.n_permute
        list_mmd2 = QuadMMDTest.permutation_list_mmd2(X, Y, wx, wy, k, repeats)
        # approximate p-value with the permutations 
        pvalue = np.mean(list_mmd2 > mmd2_stat)

        return pvalue
    
    def compute_stat(self, tst_data, use_1sample_U=True):
        """Compute the test statistic: empirical quadratic MMD^2"""
        X, Y = tst_data.xy()
        wx, wy = tst_data.weights()
        nx = X.shape[0]
        ny = Y.shape[0]

        if nx != ny:
            raise ValueError('nx must be the same as ny')

        k = self.kernel
        mmd2, var = QuadMMDTest.h1_mean_var(X, Y, wx, wy, k, is_var_computed=False,
                use_1sample_U=use_1sample_U)
        return mmd2

    @staticmethod 
    def permutation_list_mmd2(X, Y, wx, wy, k, n_permute=400, seed=8273):
        """
        Repeatedly mix, permute X,Y and compute MMD^2. This is intended to be
        used to approximate the null distritubion.
        TODO: This is a naive implementation where the kernel matrix is recomputed 
        for each permutation. We might be able to improve this if needed.
        """
        return QuadMMDTest.permutation_list_mmd2_gram(X, Y, wx, wy, k, n_permute, seed)

    @staticmethod 
    def permutation_list_mmd2_gram(X, Y, wx, wy, k, n_permute=400, seed=8273):
        """
        Repeatedly mix, permute X,Y and compute MMD^2. This is intended to be
        used to approximate the null distritubion.
        """
        XY = np.vstack((X, Y))
        wxy = np.vstack((wx,wy))
        Kxyxy = k.eval(XY, XY)#np.multiply(np.outer(wxy,wxy),k.eval(XY, XY))

        rand_state = np.random.get_state()
        np.random.seed(seed)

        nxy = XY.shape[0]#nxy = np.sum(wxy)#
        nx = X.shape[0]#nx = np.sum(wx)#
        y = Y.shape[0]#ny= np.sum(wy)#n
        list_mmd2 = np.zeros(n_permute)

        for r in range(n_permute):
            #print r
            ind = np.random.choice(nxy, nxy, replace=False)#len(wxy), len(wxy)
            # divide into new X, Y
            indx = ind[:nx]
            #print(indx)
            indy = ind[nx:]
            Kx = Kxyxy[np.ix_(indx, indx)]
            #print(Kx)
            Ky = Kxyxy[np.ix_(indy, indy)]
            Kxy = Kxyxy[np.ix_(indx, indy)]

            mmd2r, var = QuadMMDTest.h1_mean_var_gram(Kx, Ky, Kxy, wx, wy, is_var_computed=False)
            list_mmd2[r] = mmd2r

        np.random.set_state(rand_state)
        return list_mmd2

    @staticmethod
    def h1_mean_var_gram(Kx, Ky, Kxy, wx, wy, is_var_computed, use_1sample_U=True):
        """
        Same as h1_mean_var() but takes in Gram matrices directly.
        """

        nx = Kx.shape[0] #nx = np.sum(wx)#
        ny = Ky.shape[0] # ny = np.sum(wy)#
        xx = old_div((np.sum(Kx) - np.sum(np.diag(Kx))),(nx*(nx - 1)))#xx = old_div(np.sum(Kx),(nx*nx))#
        yy = old_div((np.sum(Ky) - np.sum(np.diag(Ky))),(ny*(ny - 1)))#yy = old_div(np.sum(Ky),(ny*ny))#
        # one-sample U-statistic.
        if use_1sample_U:
            xy = old_div((np.sum(Kxy) - np.sum(np.diag(Kxy))),(nx*(ny-1)))
        else:
            xy = old_div(np.sum(Kxy),(nx*ny))
        mmd2 = xx - 2*xy + yy

        if not is_var_computed:
            return mmd2, None

        # compute the variance
        Kxd = Kx - np.diag(np.diag(Kx))
        Kyd = Ky - np.diag(np.diag(Ky))
        m = nx 
        n = ny
        v = np.zeros(11)

        Kxd_sum = np.sum(Kxd)
        Kyd_sum = np.sum(Kyd)
        Kxy_sum = np.sum(Kxy)
        Kxy2_sum = np.sum(Kxy**2)
        Kxd0_red = np.sum(Kxd, 1)
        Kyd0_red = np.sum(Kyd, 1)
        Kxy1 = np.sum(Kxy, 1)
        Kyx1 = np.sum(Kxy, 0)

        #  varEst = 1/m/(m-1)/(m-2)    * ( sum(Kxd,1)*sum(Kxd,2) - sum(sum(Kxd.^2)))  ...
        v[0] = 1.0/m/(m-1)/(m-2)*( np.dot(Kxd0_red, Kxd0_red ) - np.sum(Kxd**2) )
        #           -  (  1/m/(m-1)   *  sum(sum(Kxd))  )^2 ...
        v[1] = -( 1.0/m/(m-1) * Kxd_sum )**2
        #           -  2/m/(m-1)/n     *  sum(Kxd,1) * sum(Kxy,2)  ...
        v[2] = -2.0/m/(m-1)/n * np.dot(Kxd0_red, Kxy1)
        #           +  2/m^2/(m-1)/n   * sum(sum(Kxd))*sum(sum(Kxy)) ...
        v[3] = 2.0/(m**2)/(m-1)/n * Kxd_sum*Kxy_sum
        #           +  1/(n)/(n-1)/(n-2) * ( sum(Kyd,1)*sum(Kyd,2) - sum(sum(Kyd.^2)))  ...
        v[4] = 1.0/n/(n-1)/(n-2)*( np.dot(Kyd0_red, Kyd0_red) - np.sum(Kyd**2 ) ) 
        #           -  ( 1/n/(n-1)   * sum(sum(Kyd))  )^2	...		       
        v[5] = -( 1.0/n/(n-1) * Kyd_sum )**2
        #           -  2/n/(n-1)/m     * sum(Kyd,1) * sum(Kxy',2)  ...
        v[6] = -2.0/n/(n-1)/m * np.dot(Kyd0_red, Kyx1)

        #           +  2/n^2/(n-1)/m  * sum(sum(Kyd))*sum(sum(Kxy)) ...
        v[7] = 2.0/(n**2)/(n-1)/m * Kyd_sum*Kxy_sum
        #           +  1/n/(n-1)/m   * ( sum(Kxy',1)*sum(Kxy,2) -sum(sum(Kxy.^2))  ) ...
        v[8] = 1.0/n/(n-1)/m * ( np.dot(Kxy1, Kxy1) - Kxy2_sum )
        #           - 2*(1/n/m        * sum(sum(Kxy))  )^2 ...
        v[9] = -2.0*( 1.0/n/m*Kxy_sum )**2
        #           +   1/m/(m-1)/n   *  ( sum(Kxy,1)*sum(Kxy',2) - sum(sum(Kxy.^2)))  ;
        v[10] = 1.0/m/(m-1)/n * ( np.dot(Kyx1, Kyx1) - Kxy2_sum )


        #%additional low order correction made to some terms compared with ICLR submission
        #%these corrections are of the same order as the 2nd order term and will
        #%be unimportant far from the null.

        #   %Eq. 13 p. 11 ICLR 2016. This uses ONLY first order term
        #   varEst = 4*(m-2)/m/(m-1) *  varEst  ;
        varEst1st = 4.0*(m-2)/m/(m-1) * np.sum(v)

        Kxyd = Kxy - np.diag(np.diag(Kxy))
        #   %Eq. 13 p. 11 ICLR 2016: correction by adding 2nd order term
        #   varEst2nd = 2/m/(m-1) * 1/n/(n-1) * sum(sum( (Kxd + Kyd - Kxyd - Kxyd').^2 ));
        varEst2nd = 2.0/m/(m-1) * 1/n/(n-1) * np.sum( (Kxd + Kyd - Kxyd - Kxyd.T)**2)

        #   varEst = varEst + varEst2nd;
        varEst = varEst1st + varEst2nd

        #   %use only 2nd order term if variance estimate negative
        if varEst<0:
            varEst =  varEst2nd
        return mmd2, varEst

    @staticmethod
    def h1_mean_var(X, Y, wx, wy, k, is_var_computed, use_1sample_U=True):
        """
        X: nxd numpy array 
        Y: nxd numpy array
        k: a Kernel object 
        is_var_computed: if True, compute the variance. If False, return None.
        use_1sample_U: if True, use one-sample U statistic for the cross term 
          i.e., k(X, Y).
        Code based on Arthur Gretton's Matlab implementation for
        Bounliphone et. al., 2016.
        return (MMD^2, var[MMD^2]) under H1
        """

        Kx = k.eval(X,X)#Kx = np.multiply(np.outer(wx,wx),k.eval(X, X))#
        Ky = k.eval(Y,Y)#Ky = np.multiply(np.outer(wy,wy),k.eval(Y,Y))
        Kxy = k.eval(X,Y)#Kxy = np.multiply(np.outer(wx,wy),k.eval(X, Y))#

        return QuadMMDTest.h1_mean_var_gram(Kx, Ky, Kxy, wx, wy, is_var_computed, use_1sample_U)

    @staticmethod
    def grid_search_kernel(tst_data, list_kernels, alpha, reg=1e-3):
        """
        Return from the list the best kernel that maximizes the test power criterion.
        
        In principle, the test threshold depends on the null distribution, which 
        changes with kernel. Thus, we need to recompute the threshold for each kernel
        (require permutations), which is expensive. However, asymptotically 
        the threshold goes to 0. So, for each kernel, the criterion needed is
        the ratio mean/variance of the MMD^2. (Source: Arthur Gretton)
        This is an approximate to avoid doing permutations for each kernel 
        candidate.
        - reg: regularization parameter
        return: (best kernel index, list of test power objective values)
        """
        import time
        X, Y = tst_data.xy()
        wx, wy = tst_data.weights()
        n = X.shape[0]
        obj_values = np.zeros(len(list_kernels))
        for ki, k in enumerate(list_kernels):
            start = time.time()
            mmd2, mmd2_var = QuadMMDTest.h1_mean_var(X, Y, wx, wy, k, is_var_computed=True)# 
            obj = float(mmd2)/((mmd2_var + reg)**0.5)
            obj_values[ki] = obj
            end = time.time()
            #print('(%d/%d) %s: mmd2: %.3g, var: %.3g, power obj: %g, took: %s'%(ki+1,
            #    len(list_kernels), str(k), mmd2, mmd2_var, obj, end-start))
        best_ind = np.argmax(obj_values)
        return best_ind, obj_values
    

    
def RMMD(t1,y1,t2,y2,alpha=0.01,output = 'stat'):
    '''
    Runs full test with all optimization procedures included
    output: the desired output, one of 'stat', 'p_value', 'full'
    '''
    
    if t1 == None:
        # generate random features
        x1 = general_utils.generate_random_features_1d(y1,num_feat = 100) + np.mean(y1)/np.var(y1)
        x2 = general_utils.generate_random_features_1d(y2,num_feat = 100) + np.mean(y2)/np.var(y2)
    else:
        # generate random features
        x1 = general_utils.generate_random_features(t1,y1,num_feat = 100)
        x2 = general_utils.generate_random_features(t2,y2,num_feat = 100)
    
    # define training and testing sets
    w1, w2 = [len(t) for t in y1], [len(t) for t in y2]
    tst_data = data.TSTData(x1, x2, w1, w2)
    tr, te = tst_data.split_tr_te(tr_proportion=0.5, seed=10)
    xtr, ytr = tr.xy()
    xytr = tr.stack_xy()
    #sig2 = general_utils.meddistance(xytr, subsample=1000)
    #k = kernel_utils.KGauss(sig2)
    
    # choose the best parameter and perform a test with permutations
    med = general_utils.meddistance(tr.stack_xy(), 1000)
    list_gwidth = np.hstack( ( (med**2) *(2.0**np.linspace(-3, 3, 10) ) ) )
    list_gwidth.sort()

    list_kernels = [kernel_utils.KGauss(gw2) for gw2 in list_gwidth]

    # grid search to choose the best Gaussian width
    besti, powers = QuadMMDTest.grid_search_kernel(tr, list_kernels, alpha)
    # perform test 
    best_ker = list_kernels[besti]
    
    mmd_test = QuadMMDTest(best_ker, n_permute=200, alpha=alpha)
    if output == 'stat':
        return mmd_test.compute_stat(te)
    if output == 'p_value':
        return mmd_test.compute_pvalue(te)
    if output == 'full':
        return mmd_test.perform_test(te)


# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

def MMD(t1,y1,t2,y2,alpha=0.01,output = 'stat'):
    '''
    Runs full test with all optimization procedures included
    output: the desired output, one of 'stat', 'p_value', 'full'
    '''
    
    # interpolate
    t1, x1 = interpolate(t1,y1)
    t2, x2 = interpolate(t2,y2)
    
    # define training and testing sets
    w1, w2 = [len(t) for t in t1], [len(t) for t in t2]
    tst_data = data.TSTData(x1, x2, w1, w2)
    tr, te = tst_data.split_tr_te(tr_proportion=0.5, seed=10)
    xtr, ytr = tr.xy()
    xytr = tr.stack_xy()
    sig2 = general_utils.meddistance(xytr, subsample=1000)
    k = kernel_utils.KGauss(sig2)
    
    # choose the best parameter and perform a test with permutations
    med = general_utils.meddistance(tr.stack_xy(), 1000)
    list_gwidth = np.hstack( ( (med**2) *(2.0**np.linspace(-4, 4, 20) ) ) )
    list_gwidth.sort()

    list_kernels = [kernel_utils.KGauss(gw2) for gw2 in list_gwidth]

    # grid search to choose the best Gaussian width
    besti, powers = QuadMMDTest.grid_search_kernel(tr, list_kernels, alpha)
    # perform test 
    best_ker = list_kernels[besti]
    
    mmd_test = QuadMMDTest(best_ker, n_permute=200, alpha=alpha)
    if output == 'stat':
        return mmd_test.compute_stat(te)
    if output == 'p_value':
        return mmd_test.compute_pvalue(te)
    if output == 'full':
        return mmd_test.perform_test(te)
    
def interpolate(t,y,num_obs=5):
    """
    Interpolates each trajectory with a cubic function such that observation times coincide for each one
    """
    if isinstance(t,list):
        t, y = np.array(t), np.array(y)
        
    t = [np.insert(t[i], 0, 0, axis=0) for i in range(len(t))]
    t = [np.insert(t[i], len(t[i]), 1, axis=0) for i in range(len(t))] 
    y = [np.insert(y[i], 0, y[i][0], axis=0) for i in range(len(y))] 
    y = [np.insert(y[i], len(y[i]), y[0][-1], axis=0) for i in range(len(y))]
    
    new_t = np.zeros(num_obs)
    new_y = np.zeros(num_obs)
    
    for i in range(len(t)):
        f = interp1d(t[i], y[i], kind='cubic')
        t_temp = np.linspace(0.1, 0.9, num=num_obs, endpoint=True)
        y_temp = f(t_temp)
        new_y = np.vstack((new_y, y_temp))
        new_t = np.vstack((new_t, t_temp))
        
    return new_t[1:], new_y[1:]

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

def train_test_split(T1,X1,T2,X2,train_rate=0.8):
    """
    :param train_rate: fraction of data used for training
    :param parameters: specification for the data generation of two scenarios
    :return:training and testing data for C2ST, note each is a combination of data from two samples
    """

    # %% Data Preprocessing
    # interpolate
    T1, X1 = interpolate(T1,X1)
    T2, X2 = interpolate(T2,X2)
    dataX1 = np.zeros((X1.shape[0],X1.shape[1],2))
    dataX2 = np.zeros((X2.shape[0], X2.shape[1], 2))


    # Dataset build
    for i in range(len(X1)):
        dataX1[i,:,:] = np.hstack((X1[i,np.newaxis].T,T1[i,np.newaxis].T))
        dataX2[i, :, :] = np.hstack((X2[i, np.newaxis].T, T2[i, np.newaxis].T))

    dataY1 = np.random.choice([0],size=(len(dataX1),));    dataY2 = np.random.choice([1],size=(len(dataX2),))
    dataY1 = dataY1[:,np.newaxis];    dataY2 = dataY2[:,np.newaxis]

    dataX = Permute(np.vstack((dataX1,dataX2)))
    dataY = Permute(np.vstack((dataY1,dataY2)))

    # %% Train / Test Division
    train_size = int(len(dataX) * train_rate)

    trainX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:len(dataX)])
    trainY, testY = np.array(dataY[0:train_size]), np.array(dataY[train_size:len(dataX)])

    return trainX, trainY, testX, testY

def Permute(x,seed=1):
    np.random.seed(seed)
    n = len(x)
    idx = np.random.permutation(n)
    out = x[idx]
    return out


# TODO make this a class
def C2ST(t1,y1,t2,y2,output='p_value',train_rate = 0.5):
    """
    Classifier two sample test following the procedure from (Lopez-Paz et al, 2017). We train an RNN that takes
    irregular time points and observations at these points to return a prediction of underlying sampling population

    Note: it requires an equal number of observations in each trajectory but observation times can be arbitrary

    Return:p-value for the hypothesis that two samples were generated from the same underlying stochastic process
    """
    
    # 3. Data Loading
    trainX, trainY, testX, testY = train_test_split(t1,y1,t2,y2,train_rate)

    # %% Main Function
    # 1. Graph Initialization
    ops.reset_default_graph()

    # 2. Parameters
    seq_length = len(trainX[0, :, 0])
    input_size = len(trainX[0, 0, :])
    target_size = len(trainY[0, :])

    learning_rate = 0.01
    iterations = 500
    hidden_layer_size = 10
    batch_size = 64

    # 3. Weights and Bias
    Wr = tf.Variable(tf.zeros([input_size, hidden_layer_size]))
    Ur = tf.Variable(tf.zeros([hidden_layer_size, hidden_layer_size]))
    br = tf.Variable(tf.zeros([hidden_layer_size]))

    Wu = tf.Variable(tf.zeros([input_size, hidden_layer_size]))
    Uu = tf.Variable(tf.zeros([hidden_layer_size, hidden_layer_size]))
    bu = tf.Variable(tf.zeros([hidden_layer_size]))

    Wh = tf.Variable(tf.zeros([input_size, hidden_layer_size]))
    Uh = tf.Variable(tf.zeros([hidden_layer_size, hidden_layer_size]))
    bh = tf.Variable(tf.zeros([hidden_layer_size]))

    # Weights for Attention
    Wa1 = tf.Variable(tf.random.truncated_normal([hidden_layer_size + input_size, hidden_layer_size], mean=0, stddev=.01))
    Wa2 = tf.Variable(tf.random.truncated_normal([hidden_layer_size, target_size], mean=0, stddev=.01))
    ba1 = tf.Variable(tf.random.truncated_normal([hidden_layer_size], mean=0, stddev=.01))
    ba2 = tf.Variable(tf.random.truncated_normal([target_size], mean=0, stddev=.01))

    # Weights for output layers
    Wo = tf.Variable(tf.random.truncated_normal([hidden_layer_size, target_size], mean=0, stddev=.01))
    bo = tf.Variable(tf.random.truncated_normal([target_size], mean=0, stddev=.01))

    # 4. Place holder
    # Target
    Y = tf.compat.v1.placeholder(tf.float32, [None, 1])
    # Input vector with shape[batch, seq, embeddings]
    _inputs = tf.compat.v1.placeholder(tf.float32, shape=[None, None, input_size], name='inputs')


    # Function to convert batch input data to use scan ops of tensorflow.
    def process_batch_input_for_RNN(batch_input):
        batch_input_ = tf.transpose(batch_input, perm=[2, 0, 1])
        X = tf.transpose(batch_input_)

        return X


    # Processing inputs to work with scan function
    processed_input = process_batch_input_for_RNN(_inputs)

    # Initial Hidden States
    initial_hidden = _inputs[:, 0, :]
    initial_hidden = tf.matmul(initial_hidden, tf.zeros([input_size, hidden_layer_size]))


    # 5. Function for Forward GRU cell.
    def GRU(previous_hidden_state, x):
        # R Gate
        r = tf.sigmoid(tf.matmul(x, Wr) + tf.matmul(previous_hidden_state, Ur) + br)

        # U Gate
        u = tf.sigmoid(tf.matmul(x, Wu) + tf.matmul(previous_hidden_state, Uu) + bu)

        # Final Memory cell
        c = tf.tanh(tf.matmul(x, Wh) + tf.matmul(tf.multiply(r, previous_hidden_state), Uh) + bh)

        # Current Hidden state
        current_hidden_state = tf.multiply((1 - u), previous_hidden_state) + tf.multiply(u, c)

        return current_hidden_state


    # 6. Function to get the hidden and memory cells after forward pass
    def get_states():
        # Getting all hidden state through time
        all_hidden_states = tf.scan(GRU, processed_input, initializer=initial_hidden, name='states')

        return all_hidden_states


    # %% Attention

    # Function to get attention with the last input
    def get_attention(hidden_state):
        inputs = tf.concat((hidden_state, processed_input[-1]), axis=1)
        hidden_values = tf.nn.tanh(tf.matmul(inputs, Wa1) + ba1)
        e_values = (tf.matmul(hidden_values, Wa2) + ba2)

        return e_values


    # Function for getting output and attention coefficient
    def get_outputs():
        all_hidden_states = get_states()

        all_attention = tf.map_fn(get_attention, all_hidden_states)

        a_values = tf.nn.softmax(all_attention, axis=0)

        final_hidden_state = tf.einsum('ijk,ijl->jkl', a_values, all_hidden_states)

        output = tf.nn.sigmoid(tf.matmul(final_hidden_state[:, 0, :], Wo) + bo)

        return output, a_values


    # Getting all outputs from rnn
    outputs, attention_values = get_outputs()

    # reshape out for sequence_loss
    loss = tf.sqrt(tf.reduce_mean(tf.square(outputs - Y)))

    # Optimization
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)

    # Sessions
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())


    # 3. Sample from the real data (Mini-batch index sampling)
    def sample_X(m, n):
        return np.random.permutation(m)[:n]


    # Training step
    for i in range(iterations):

        idx = sample_X(len(trainX[:, 0, 0]), batch_size)

        Input = trainX[idx, :, :]

        _, step_loss = sess.run([train, loss], feed_dict={Y: trainY[idx], _inputs: Input})

        #if i % 100 == 0:
        #    print("[step: {}] loss: {}".format(i, step_loss))

    # %% Evaluation
    final_outputs, final_attention_values = sess.run([outputs, attention_values], feed_dict={_inputs: testX})

    accuracy = np.mean(np.round(final_outputs)==testY)

    p_value = 1 - normal.cdf(accuracy,1/2,np.sqrt(1/(4*len(testX))))
    
    if output == 'stat':
        return accuracy
    if output == 'p_value':
        return p_value
    
#TODO: make this a class
def GP_test(t1,y1,t2,y2,output='p_value'):
    '''
    Test on fitted GPs for the two samples
    '''
    
    def compute_posterior(t,y,u=np.linspace(0,1,10),kernel='rbf'):
        '''
        Compute posterior mean and variance under GP model
        inputs:
        - times t, (n x d) array
        - observations y, (n x d) array
        - inducing points u
        - kernel: chosen family of kernels, one of ['rbf','Matern', 'RationalQuadratic','ExpSineSquared']
        outputs:
        - vector of posterior means and covariance matrix at inducing points
        '''

        
        if kernel == 'rbf':
            kernel = 1.0*RBF(length_scale_bounds=(1e-3,100.0)) + C(1.0, (1e-3, 1e3)) +\
                     WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-10, 1e+1))
        if kernel == 'Matern':
            kernel = 1.0*Matern(length_scale_bounds=(1e-3,100.0))+ \
                     WhiteKernel(noise_level=0.05, noise_level_bounds=(1e-10, 1e+1)) + C(1.0, (1e-3, 1e3))
        if kernel == 'RationalQuadratic':
            kernel = 1.0*RationalQuadratic(length_scale_bounds=(1e-1,100.0))+ \
                     WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))+ C(1.0, (1e-3, 1e3))
        if kernel == 'ExpSineSquared':
            kernel = 1.0*ExpSineSquared(length_scale_bounds=(1e-1,100.0))+ \
                     WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))+ C(1.0, (1e-3, 1e3))

        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3)
        gp.fit(np.reshape(t.flatten(),(-1,1)), y.flatten())
        mean, cov = gp.predict(np.reshape(u,(-1,1)), return_cov=True)

        return mean, cov

    # GP approximations to each sample
    mean1, cov1 = compute_posterior(t1,y1)
    mean2, cov2 = compute_posterior(t2,y2)

    # Compute statistic
    mdiff = mean2 - mean1
    cov = cov1 + cov2
    chi2_stat = np.dot(mdiff @ inv(cov),mdiff)
    #chi2_stat = np.dot(np.linalg.solve(cov, mdiff), mdiff)
    pvalue = stats.chi2.sf(chi2_stat, len(mean1))
    
    if output == 'stat':
        return chi2_stat
    if output == 'p_value':
        return pvalue
    