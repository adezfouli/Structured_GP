# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from GPy.core.domains import POSITIVE, NEGATIVE, BOUNDED
import sys 
lim_val = -np.log(sys.float_info.epsilon) 

class transformation(object):
    domain = None
    def f(self, x):
        raise NotImplementedError

    def finv(self, x):
        raise NotImplementedError

    def gradfactor(self, f):
        """ df_dx evaluated at self.f(x)=f"""
        raise NotImplementedError

    def initialize(self, f):
        """ produce a sensible initial value for f(x)"""
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

class logexp(transformation):
    domain = POSITIVE
    def f(self, x):
        return np.where(x<-lim_val, np.log(1+np.exp(-lim_val)), np.where(x>lim_val, x, np.log(1. + np.exp(x))))
    def finv(self, f):
        return np.where(f>lim_val, f, np.log(np.exp(f) - 1.))
    def gradfactor(self, f):
        return np.where(f>lim_val, 1., 1 - np.exp(-f))
    def initialize(self, f):
        if np.any(f < 0.):
            print "Warning: changing parameters to satisfy constraints"
        return np.abs(f)
    def __str__(self):
        return '(+ve)'

class negative_logexp(transformation):
    domain = NEGATIVE
    def f(self, x):
        return -logexp.f(x)
    def finv(self, f):
        return logexp.finv(-f) 
    def gradfactor(self, f):
        return -logexp.gradfactor(-f)
    def initialize(self, f):
        return -logexp.initialize(f)
    def __str__(self):
        return '(-ve)'

class logexp_clipped(logexp):
    max_bound = 1e100
    min_bound = 1e-10
    log_max_bound = np.log(max_bound)
    log_min_bound = np.log(min_bound)
    domain = POSITIVE
    def __init__(self, lower=1e-6):
        self.lower = lower
    def f(self, x):
        exp = np.exp(np.clip(x, self.log_min_bound, self.log_max_bound))
        f = np.log(1. + exp)
#         if np.isnan(f).any():
#             import ipdb;ipdb.set_trace()
        return np.clip(f, self.min_bound, self.max_bound)
    def finv(self, f):
        return np.log(np.exp(f - 1.))
    def gradfactor(self, f):
        ef = np.exp(f) # np.clip(f, self.min_bound, self.max_bound))
        gf = (ef - 1.) / ef
        return gf # np.where(f < self.lower, 0, gf)
    def initialize(self, f):
        if np.any(f < 0.):
            print "Warning: changing parameters to satisfy constraints"
        return np.abs(f)
    def __str__(self):
        return '(+ve_c)'

class exponent(transformation):
    domain = POSITIVE
    def f(self, x):
        return np.where(x<lim_val, np.where(x>-lim_val, np.exp(x), np.exp(-lim_val)), np.exp(lim_val))
    def finv(self, x):
        return np.log(x)
    def gradfactor(self, f):
        return f
    def initialize(self, f):
        if np.any(f < 0.):
            print "Warning: changing parameters to satisfy constraints"
        return np.abs(f)
    def __str__(self):
        return '(+ve)'

class negative_exponent(exponent):
    domain = NEGATIVE
    def f(self, x):
        return -exponent.f(x)
    def finv(self, f):
        return exponent.finv(-f)
    def gradfactor(self, f):
        return f
    def initialize(self, f):
        return -exponent.initialize(f) #np.abs(f)
    def __str__(self):
        return '(-ve)'

class square(transformation):
    domain = POSITIVE
    def f(self, x):
        return x ** 2
    def finv(self, x):
        return np.sqrt(x)
    def gradfactor(self, f):
        return 2 * np.sqrt(f)
    def initialize(self, f):
        return np.abs(f)
    def __str__(self):
        return '(+sq)'

class logistic(transformation):
    domain = BOUNDED
    def __init__(self, lower, upper):
        assert lower < upper
        self.lower, self.upper = float(lower), float(upper)
        self.difference = self.upper - self.lower
    def f(self, x):
        return self.lower + self.difference / (1. + np.exp(-x))
    def finv(self, f):
        return np.log(np.clip(f - self.lower, 1e-10, np.inf) / np.clip(self.upper - f, 1e-10, np.inf))
    def gradfactor(self, f):
        return (f - self.lower) * (self.upper - f) / self.difference
    def initialize(self, f):
        if np.any(np.logical_or(f < self.lower, f > self.upper)):
            print "Warning: changing parameters to satisfy constraints"
        return np.where(np.logical_or(f < self.lower, f > self.upper), self.f(f * 0.), f)
    def __str__(self):
        return '({},{})'.format(self.lower, self.upper)

