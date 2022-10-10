import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from numpy.random import choice
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

class BasePU:
    @staticmethod
    def _make_propensity_weighted_data(x,s,e,sample_weight=None):
        weights_pos = s/e
        weights_neg = (1-s) + s*(1-1/e)
        if sample_weight is not None:
            weights_pos = sample_weight*weights_pos
            weights_neg = sample_weight*weights_neg
            
        Xp = np.concatenate([x,x])
        Yp = np.concatenate([np.ones_like(s), np.zeros_like(s)])
        Wp = np.concatenate([weights_pos, weights_neg])
        return Xp, Yp, Wp

class LogisticRegressionPU(LogisticRegression, BasePU):
    def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='saga', max_iter=100, # BGB  solver was 'liblinear'
                 multi_class='ovr', verbose=0, warm_start=False, n_jobs=-1): # BGB n_jobs was '1'
        LogisticRegression.__init__(self,penalty=penalty, dual=dual, tol=tol, C=C, 
                         fit_intercept=fit_intercept,intercept_scaling=intercept_scaling,
                        class_weight=class_weight, random_state=random_state,
                        solver=solver, max_iter=max_iter, multi_class=multi_class,
                        verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)
               
        
    def fit(self, x, s, e=None, sample_weight=None, minibatch=None):
        if e is None:
            super().fit(x,s,sample_weight)
        else:
            Xp,Yp,Wp = self._make_propensity_weighted_data(x,s,e,sample_weight)
            super().fit(Xp,Yp,Wp)
            

class SGDClassifierPU(SGDClassifier, BasePU):
    def __init__(self, loss = 'log', alpha = 0.0001, penalty = 'l2', tol=0.001, n_jobs=-1, validation_fraction = 0.1, n_iter_no_change=5, warm_start = True, sgd_early_stopping=True, minibatch = 100000):
        SGDClassifier.__init__(self,loss=loss, alpha=alpha, penalty = penalty, tol=tol, n_jobs = n_jobs,validation_fraction=validation_fraction,n_iter_no_change=n_iter_no_change, warm_start=warm_start)
        self.sgd_early_stopping = sgd_early_stopping
        self.minibatch = minibatch
               
        
    def fit(self, x, s, e=None, sample_weight=None):
        
        if self.minibatch is None:
            if e is None:
                super().fit(x,s,sample_weight=sample_weight)
            else:
                Xp,Yp,Wp = self._make_propensity_weighted_data(x,s,e,sample_weight)
                super().fit(Xp,Yp,sample_weight=Wp)
                
        else:
            idx_is, idx_oos = train_test_split([i for i in range(len(s))], test_size = self.validation_fraction)
            x_train, x_val, s_train, s_val = x[idx_is], x[idx_oos], s[idx_is], s[idx_oos]
            
            if sample_weight is None:
                sample_weight_sub, sample_weight_val = None, None
            elif sample_weight is not None:
                sample_weight_train, sample_weight_val = sample_weight[idx_is], sample_weight[idx_oos]
            
            i = 0
            ct = 0
            
            best_logloss = np.inf
            
            while i < self.max_iter:
                idx = choice(len(idx_is), self.minibatch)
                x_sub, s_sub = x_train[idx], s_train[idx]
                
                if sample_weight is not None:
                    sample_weight_sub = sample_weight_train[idx]
                
                if e is None:
                    super().partial_fit(x_sub,s_sub,classes=np.unique(s),sample_weight=sample_weight_sub)
                    logloss = log_loss(s_val, super().predict_proba(x_val), sample_weight = sample_weight_val)
                else:                    
                    idx_is_rw = idx_is + list(map(lambda x: x + len(s), idx_is)) 
                    idx_oos_rw = idx_oos + list(map(lambda x: x + len(s), idx_oos)) 
                    idx_rw = np.concatenate([idx, list(map(lambda x: x + len(idx_is), idx))])
                    
                    Xp, Yp, Wp = self._make_propensity_weighted_data(x,s,e,sample_weight)      
                    Xp_sub, Yp_sub, Wp_sub, Xp_val, Yp_val, Wp_val = Xp[idx_is_rw][idx_rw], Yp[idx_is_rw][idx_rw], Wp[idx_is_rw][idx_rw], Xp[idx_oos_rw], Yp[idx_oos_rw], Wp[idx_oos_rw]
                    super().partial_fit(Xp_sub,Yp_sub,classes=np.unique(Yp),sample_weight=Wp_sub)
                    logloss = log_loss(Yp_val, super().predict_proba(Xp_val), sample_weight = Wp_val)
                    
                if self.sgd_early_stopping:
                    if logloss + self.tol < best_logloss:    
                        best_logloss = logloss
                        ct = 0
                    elif logloss + self.tol >= best_logloss:
                        ct += 1
                    i += 1
                    if ct >= self.n_iter_no_change:
                        break