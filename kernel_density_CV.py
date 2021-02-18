# https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/
# https://github.com/opendifferentialprivacy/smartnoise-sdk/blob/50f223161242d135d495ec39618414fd2ca104ed/sdk/opendp/smartnoise/evaluation/evaluator/_dp_evaluator.py#L272
# https://www.r-bloggers.com/2014/10/cross-validation-for-kernel-density-estimation/

# what we need
# input: a vector of data, a specific interval
# output: the area under the curve of kernel density estimate for this specific interval

# gaussian kernel density estimation
# from scipy.stats import gaussian_kde

# load the libraries
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV
import random

#
def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)

# generate synthetic univariate data from standard normal distribution
n = 500

# rule of thumb bandwidth
# Scott's rule: h = n**(-1./ (d+4))
# another rule of thumb that takes standard error estimate into consideration
# h = (4 \sigma^5 / 3n)^(1/5) ~ 1.06 * sigma * n^(-1/5)
random.seed(2021)

x = np.random.randn(n)
h_default = 1.06 * np.std(x) * len(x)**(-1/5)
# x.reshape((1,-1))

#
kde = KernelDensity(kernel='gaussian', bandwidth=h_default).fit(x[:,None])
# add None to make it 2D
#
# log_density = kde.score_samples(x[:3])
# figure out the log_density at a particular time point
# kde.score_samples([-1])

# X_grid = np.linspace(-3, 3, 600)
X_grid = np.array([3])
kde.score_samples(X_grid[:,None])

# numerical integration in python
import scipy.integrate as integrate
import scipy.special as special

result = integrate.quad(lambda x: special.jv(2.5,x), 0, 4.5)
# second value in result is the estimated absolute error

#
result = integrate.quad(lambda x: np.exp(kde.score_samples(np.array([x])[:,None])), 0, 0.5)



# grid search CV for finding the optimal bandwidth
grid = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.linspace(0.1, 1.0, 30)},
                    cv=20) # 20-fold cross-validation
grid.fit(x[:, None])

print(grid.best_params_)

grid.best_params_["bandwidth"]