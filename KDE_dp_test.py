####### this script is used to define a function that takes two
####### neighboring datasets as inputs and run a DP test based on
####### kernel density estimates on them
####### input: fD1, fD2, ep, pp, method for selecting bandwidth

from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

#######
def KDE_dp_test(self, fD1, fD2, bandwidth_method = 'rule-of-thumb', binlist, ep, pp):
    """
    :param self:
    :param fD1: dataset 1, needs to be an 1D numpy array
    :param fD2: dataset 2, needs to be an 1D numpy array
    :param bandwidth_method: default 'rule-of-thumb', 'cv' is also an option
    :param binlist: list of bins for comparison
    :param ep: EvaluatorParams
    :param pp: PrivacyParams
    :return:
    """

    # bandwidth selection
    if bandwidth_method == 'cv':
        hmax1 = 1.144 * np.std(fD1) * fD1.size ** (-1/5)
        hmax2 = 1.144 * np.std(fD2) * fD2.size ** (-1/5) # based on the choice in bw.ucv function in R
        grid1 = GridSearchCV(KernelDensity(),
                            {'bandwidth': np.linspace(hmax1*0.1, hmax1, 100)},
                            cv=20)  # 20-fold cross-validation
        grid2 = GridSearchCV(KernelDensity(),
                            {'bandwidth': np.linspace(hmax2*0.1, hmax2, 30)},
                            cv=20)  # 20-fold cross-validation
        # fit
        grid1.fit(fD1[:, None])
        grid2.fit(fD2[:, None])
        #
        h1 = grid1.best_params_["bandwidth"]
        h2 = grid2.best_params_["bandwidth"]
    else:
        # rule of thumb choice of bandwidth
        h1 = 1.06 * np.std(fD1) * fD1.size ** (-1/5)
        h2 = 1.06 * np.std(fD2) * fD2.size ** (-1/5)

    #
    kde1 = KernelDensity(kernel='gaussian', bandwidth=h1).fit(fD1[:, None])
    kde2 = KernelDensity(kernel='gaussian', bandwidth=h2).fit(fD2[:, None])

    # go over the binlist
    num_buckets = binlist.size - 1

    # area under curve based on numerical integration
    result1 = integrate.quad(lambda x: np.exp(kde1.score_samples(np.array([x])[:, None])), 0, 0.5)
    result2 = integrate.quad(lambda x: np.exp(kde2.score_samples(np.array([x])[:, None])), 0, 0.5)

    #