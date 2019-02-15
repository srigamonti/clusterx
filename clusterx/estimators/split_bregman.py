# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.

from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
from copy import deepcopy

class SplitBregman(BaseEstimator, RegressorMixin):
    """**Split Bregmann Iteration class**

    the split bregman iteration selects a sparse solution of ECIs in the cluster expansion construction
    given the ab initio energy (evals) and correlation matrix (corr)

    **Parameters:**

    Parameters for split bregman include lambda and mu_min, mu_max, and mu_step

    ``mu``: sparsity hyperparameter, integer
    ``lamb``: coupling parameter, integer
    ``tol``: optional tolerance level for convergance of the algorithm
    ``mult``: list of cluster multiplicities

    """
    def __init__(self, mult=None, lamb=0.9, mu=1.0, estimator_type="regressor",tol=1.0e-10):
        self.mu = mu
        self.lamb = lamb
        self.tol = tol
        self.mult = mult
        self._estimator_type = "regressor"
        # self.ecis = []
        # self.coef_ = []
        # self.ergs = []
        # self.X_ = []
        # self.Y_ = []

    def fit(self, corr, evals):
        """Train the model on data"""
        self.X_ = deepcopy(corr)
        self.Y_ = deepcopy(evals)
        Ncl = np.shape(self.mult)[0]
        lastd = np.zeros((Ncl))
        nextd = np.zeros((Ncl))
        lastb = np.ones((Ncl))
        nextb = np.zeros((Ncl))
        last_eci = np.zeros((Ncl))
        diff = 1
        while diff > float(self.tol):
            next_eci = self._split_bregman_rms_opt(self.X_, self.Y_, nextb, lastd, shrink_threshold = 0.1*min(evals))
            self.ecis = deepcopy(next_eci)
            next_eci_mu_prod = np.dot(self.ecis, self.mu)
            nextd = self._split_bregman_shrink(next_eci_mu_prod, nextb)
            tmp = np.subtract(next_eci_mu_prod, nextd)
            lastb = nextb
            nextb = np.add(lastb, tmp)
            lastd = nextd
            diff = float(sum(np.subtract(last_eci, next_eci)))
            last_eci = deepcopy(next_eci)

        self.coef_ = deepcopy(next_eci)

        return self

    def predict(self, corr):
        """predict the model on data"""
        ecis = deepcopy(self.ecis)
        mults = deepcopy(self.mult)
        ncl = len(mults)
        elist = np.zeros(corr.shape[0])
        nstruc = corr.shape[0]
        for n in range(nstruc):
            erg=0
            for j in range(ncl):
                erg += mults[j] * ecis[j] * corr[n,j]
            elist[n] = erg

        self.ergs = deepcopy(elist)

        return self.ergs

    def _split_bregman_rms_opt(self, corr, evals, b, d, shrink_threshold = None):
        N = np.shape(evals)
        Ncl = np.shape(self.mult)[0]
        corr_trans = corr.T
        corr_prod = np.dot(corr_trans, corr)
        lambmu2_matrix = np.zeros((Ncl, Ncl))
        np.fill_diagonal(lambmu2_matrix, float(self.mu)**2*float(self.lamb))
        sum_XtX_lambmu2 = np.add(corr_prod, lambmu2_matrix)
        lambmu_matrix = np.zeros((Ncl, Ncl))
        np.fill_diagonal(lambmu_matrix, float(self.mu)**2*float(self.lamb))
        sub_b_d = np.subtract(d,b)
        lambmu_identy_sub_b_d = np.dot(lambmu_matrix,sub_b_d)
        XtE = np.dot(corr_trans, evals)
        first_term = np.linalg.solve(sum_XtX_lambmu2,XtE)
        second_term = np.linalg.solve(sum_XtX_lambmu2,lambmu_identy_sub_b_d)
        ecimult = first_term + second_term
        eci = []
        for i in range(np.shape(ecimult)[0]):
            eci.append(ecimult[i] / (1.0*float(self.mult[i])))
            if shrink_threshold is not None:
                if abs(eci[i]) < shrink_threshold:
                    eci[i] = 0.0

        return np.array(eci)

    def _split_bregman_shrink(self, eci, b):
        combine = np.add(eci, b)
        nextd = np.zeros(np.shape(combine)[0])
        for i, x in enumerate(combine):
            if abs(float(x)) <= float(1/float(self.lamb)):
                nextd[i] = 0.0
            else:
                if float(x) > float(1/float(self.lamb)):
                    nextd[i] = (float(x)/abs(float(x)))*float(x - 1/float(self.lamb))
                elif float(x) < float(-1/float(self.lamb)):
                    nextd[i] = (float(x)/abs(float(x)))*float(x + 1/float(self.lamb))

        return nextd

    def split_bregman_eval_energy(self, ecisE, multE, corr):

        erg = 0
        for j in range(len(ecisE)):
            erg += multE[j] * ecisE[j] * corr[j]

        return erg
