
from utils import open_file
import numpy as np
import time

import re
import pandas as pd
# from mlutils import dataset, connector
import scipy.stats
from scipy.stats import *
from sklearn.preprocessing import StandardScaler
import math
import matplotlib.pyplot as plt
import warnings
# import statsmodels.api as sm
import seaborn as sns
import pylab as py

warnings.filterwarnings('ignore')




def get_good_indices(name=None):
    """
    Returns indices of bands which are not noisy

    Parameters:
    ---------------------
    name: name
    Returns:
    -----------------------
    numpy array of good indices
    """
    indices = np.arange(128)
    indices = indices[5:-7]
    indices = np.delete(indices, [43, 44, 45])
    return indices

CUSTOM_DATASETS_CONFIG = {
    'DFC2018_HSI': {
        'img': '2018_IEEE_GRSS_DFC_HSI_TR.HDR',
        'gt': '2018_IEEE_GRSS_DFC_GT_TR.tif',
        'download': False,
        'loader': lambda folder: dfc2018_loader(folder)
    },
    # 'RApple': {
    #     'urls': ['http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
    #              'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'],
    #     'img': 'RApple_3000.mat',
    #     'gt': 'RApple_gt.mat',
    #     'download': False,
    #     'loader': lambda folder: rapple_loader(folder)
    #     },
    'RApple': {
        'urls': ['http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
                 'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'],
        'img': 'RApple_3000.hdr',
        'gt': 'RApple_gt.mat',
        'download': False,
        'loader': lambda folder: rapple_loader(folder)
    },
    'Depth': {
        'urls': 'https://doi.org/10.5281/zenodo.3984905',
        'img': 'led.tif',
        'calibr': 'hg1.tif',
        'limit': 60,
        # 'gt': 'F_1.npz',
        'download': False,
        'loader': lambda folder: depth_loader(folder)
    },
    'Blood': {
        'urls': 'https://doi.org/10.5281/zenodo.3984905',
        'img': 'F_1.hdr',
        'gt': 'F_1.npz',
        'download': False,
        'loader': lambda folder: blood_loader(folder)
    }
}

def blood_loader(folder):
    palette = None
    img = open_file(folder + 'F_1.hdr')  # [:, :, :-2]
    # removal of damaged sensor line
    img = np.delete(img, 445, 0)
    img = img[:, :, get_good_indices()]  # [:, :, get_good_indices(name)]
    gt = open_file(folder + 'F_1.npz')
    # gt = gt.astype('uint8')

    rgb_bands = (47, 31, 15)

    label_values = ["background",
                    "blood",
                    "ketchup",
                    "artificial blood",
                    "beetroot juice",
                    "poster paint",
                    "tomato concentrate",
                    "acrtylic paint",
                    "uncertain blood"]
    ignored_labels = [0]
    return img, gt, rgb_bands, ignored_labels, label_values, palette

def depth_loader(folder):

    def sum_lines() :
        #all lines sum data
        res=[]
        for j in range(0, 1535):  # 1536
            res.append((j,
                       np.amax(img[j]), # abs max
                       np.sum(img[j])/200
                        ))
        res = np.array(res)
        np.savetxt(folder + "sum" + CUSTOM_DATASETS_CONFIG['Depth']['img'] + ".txt", res, fmt='%i', delimiter=',')

    # lines with data

    # it = np.nditer(indices, flags=['f_index'])
    #
    # res = []
    # for j in it:
    #     row = img[j]
    #     max_index = np.argmax(row)
    #     print(j, " ", max_index, " ", row[max_index])
    #     distr = guess_distribution(img, j)
    #     res.append((j,
    #                 np.amax(img[j]), #abs max
    #                 distr[1][0], #median
    #                 distr[1][0],  # mean
    #                 distr[1][2][0][0]#, #mode
    #                 distr[0], distr[1], distr[2], distr[3], distr[4]
    #                 ))
    #
    # res = np.array(res)
    # np.savetxt(folder + "distr_object" + CUSTOM_DATASETS_CONFIG['Depth']['img'] + ".txt", res, fmt='%i', delimiter=',')
    def distr_empty_lines() :
    # empty lines
        it = np.nditer(indices, flags=['f_index'])
        res = []
        for i in range(0, 200):  # 1536
            if i not in it:
                distr = guess_distribution(img, i)
                res.append((i,
                            np.amax(img[i]),  # abs max
                            distr[1][0],  # median
                            distr[1][0],  # mean
                            distr[1][2][0][0],  # mode
                            distr[0][0], distr[0][1], distr[0][2], distr[0][3], distr[0][4]
                            ))

        res = np.array(res)
        np.savetxt(folder + "distr_empty" + CUSTOM_DATASETS_CONFIG['Depth']['img'] + ".txt", res, fmt='%i', delimiter=',')
        print(res)
    palette = None
    img = open_file(folder + CUSTOM_DATASETS_CONFIG['Depth']['img'])  # [:, :, :-2]
    calibr = np.mean(open_file(folder + CUSTOM_DATASETS_CONFIG['Depth']['calibr']), axis=0)

    # removal of damaged sensor line
    #img[1123] = img[1120, 0]
    # img = np.delete(img, 1122, 0)
    # img = img[:, :, get_good_indices()]  # [:, :, get_good_indices(name)]
    max_index = np.unravel_index(img.argmax(), img.shape)
    max = img[max_index]
    # (слева примерный номер канала, справа примерная длина волны
    # 86 - 409
    # 143 - 436
    # 256 - 507
    # 349 - 551
    # 423 - 578
    # 815 - 767
    oldMax = 349
    oldMin = 86

    newMax = 551
    newMin = 409
    oldRange = oldMax - oldMin
    newRange = newMax - newMin

    oldValue = 423
    newValue = ((oldValue - oldMin) * newRange / oldRange) + newMin
    print(newValue)

    print("File ", CUSTOM_DATASETS_CONFIG['Depth']['img'], "has max value=", max, "located at ", max_index)
    img = img.astype('uint8')
    np.savetxt(CUSTOM_DATASETS_CONFIG['Depth']['img'] + ".txt", img, fmt='%i', delimiter=',')
    print("Line Position MaxValue")
    indices = list(np.where((img > CUSTOM_DATASETS_CONFIG['Depth']['limit']).any(axis=1)))
    sum_lines()
    # indices = indices[4:]
    # indices = np.delete(indices,
    # indices = np.delete( indices,
    # img = img[:, :, get_good_indices()]  # [:, :, get_good_indices(name)]

    # print("File ", CUSTOM_DATASETS_CONFIG['Depth']['img'],
    #       "has these lines  with at least one element bigger than " ,
    #       CUSTOM_DATASETS_CONFIG['Depth']['limit'], "\n",
    #       ''.join(str(e) for e in indices).replace("  ",'\n'))

    ind = np.argwhere(img == np.amax(img, 1, keepdims=True))
    print(list(map(tuple, ind)))

    gt = np.zeros((1536, 2048), dtype=int)
    # open_file(folder + 'F_1.npz')
    for i in range(650, 750):
        for j in range(750, 850):
            gt[i, j] = 1
    gt = gt.astype('uint8')

    rgb_bands = (47, 31, 15)

    label_values = ["background",
                    "blood",
                    "ketchup",
                    "artificial blood",
                    "beetroot juice",
                    "poster paint",
                    "tomato concentrate",
                    "acrtylic paint",
                    "uncertain blood"]
    ignored_labels = [0]

    return img, gt, rgb_bands, ignored_labels, label_values, palette


def guess_distribution(img, line):
    def get_max(line):
        median = np.median(img[line])
        mean = np.mean(img[line])
        mode = scipy.stats.mode(df[line])
        return [median, mean, mode]

    def standarise(column, pct, pct_lower):
        sc = StandardScaler()
        y = df[column][df[column].notnull()].to_list()
        y.sort()
        len_y = len(y)
        y = y[int(pct_lower * len_y):int(len_y * pct)]
        len_y = len(y)
        yy = ([[x] for x in y])
        sc.fit(yy)
        y_std = sc.transform(yy)
        y_std = y_std.flatten()
        return y_std, len_y, y

    def fit_distribution(column, pct, pct_lower):
        # Set up list of candidate distributions to use
        # See https://docs.scipy.org/doc/scipy/reference/stats.html for more
        y_std, size, y_org = standarise(column, pct, pct_lower)

        dist_names = ['alpha', 'anglit', 'argus', 'betaprime', 'bradford', 'burr', 'burr12', 'cauchy', 'chi', 'chi2',
                      'cosine', 'crystalball', 'dgamma', 'dweibull', 'erlang', 'expon',
                      'exponnorm', 'exponweib', 'exponpow', 'f', 'fatiguelife', 'fisk', 'foldcauchy', 'foldnorm',
                      'genlogistic', 'gennorm', 'genpareto', 'genexpon', 'genextreme', 'gausshyper', 'gamma',
                      'gengamma', 'genhalflogistic', 'geninvgauss', 'gilbrat', 'gompertz',
                      'gumbel_r', 'gumbel_l', 'halfcauchy', 'halflogistic', 'halfnorm', 'halfgennorm', 'hypsecant',
                      'invgamma', 'invgauss', 'invweibull', 'johnsonsb', 'johnsonsu',
                      'laplace', 'laplace_asymmetric', 'levy', 'levy_l', 'logistic', 'loggamma', 'loglaplace',
                      'lognorm', 'loguniform', 'lomax', 'maxwell', 'mielke', 'moyal', 'nakagami',
                      'ncx2', 'ncf', 'nct', 'norm', 'norminvgauss', 'pareto', 'pearson3', 'powerlaw', 'powerlognorm',
                      'powernorm', 'rdist', 'rayleigh', 'rice', 'recipinvgauss', 'semicircular',
                      'skewnorm', 't', 'trapezoid', 'triang', 'truncexpon', 'truncnorm', 'tukeylambda', 'uniform',
                      'vonmises', 'vonmises_line', 'wald', 'weibull_min', 'weibull_max', 'wrapcauchy']
        #dist_names = ['laplace_asymmetric', 'argus', 'gumbel_l', 'norminvgauss', 'exponnorm']
        # dist_names = ['weibull_min', 'norm', 'weibull_max']
        # 'kappa4', 'kstwo','kstwobign',
        # dist_names = [   # skip 'kappa3', 'ksone','levy_stable'
        chi_square_statistics = []
        # 11 bins
        percentile_bins = np.linspace(0, 100, 11)
        percentile_cutoffs = np.percentile(y_std, percentile_bins)
        observed_frequency, bins = (np.histogram(y_std, bins=percentile_cutoffs))
        cum_observed_frequency = np.cumsum(observed_frequency)

        # Loop through candidate distributions

        for distribution in dist_names:
            # Set up distribution and get fitted distribution parameters
            dist = getattr(scipy.stats, distribution)
            param = dist.fit(y_std)
            # print("{}\n{}\n".format(dist, param))

            # Get expected counts in percentile bins
            # cdf of fitted sistrinution across bins
            cdf_fitted = dist.cdf(percentile_cutoffs, *param)
            expected_frequency = []
            for bin in range(len(percentile_bins) - 1):
                expected_cdf_area = cdf_fitted[bin + 1] - cdf_fitted[bin]
                expected_frequency.append(expected_cdf_area)

            # Chi-square Statistics
            expected_frequency = np.array(expected_frequency) * size
            cum_expected_frequency = np.cumsum(expected_frequency)
            ss = round(sum(((cum_expected_frequency - cum_observed_frequency) ** 2) / cum_observed_frequency), 0)
            chi_square_statistics.append(ss)

        # Sort by minimum ch-square statistics
        results = pd.DataFrame()
        results['Distribution'] = dist_names
        results['chi_square'] = chi_square_statistics
        results.sort_values(['chi_square'], inplace=True)

        print('\nDistributions listed by Betterment of fit:')
        print('............................................')
        print(results.head(5))

        # dist = getattr(scipy.stats, dist_names[results.index[0]])
        # mean = dist.mean(y_org)
        return results.index[0:5]

    df = pd.DataFrame(img)
    df = df.transpose()
    distr = fit_distribution(line, 0.99, 0.01)

    print("distrib for line #", line, " = ", distr)
    return [distr, get_max(line)]
    # print(df.head())
    # print(df.columns)


def rapple_loader(folder):
    palette = None
    img = open_file(folder + 'RApple_3000.hdr')[:, :, :-2]
    gt = open_file(folder + 'RApple_gt.mat')['indian_pines_gt']
    # gt = gt.astype('uint8')

    rgb_bands = (47, 31, 15)

    label_values = ["Unclassified",
                    "Healthy grass",
                    "Stressed grass",
                    "Artificial turf",
                    "Evergreen trees",
                    "Deciduous trees",
                    "Bare earth",
                    "Water",
                    "Residential buildings",
                    "Non-residential buildings",
                    "Roads",
                    "Sidewalks",
                    "Crosswalks",
                    "Major thoroughfares",
                    "Highways",
                    "Railways",
                    "Paved parking lots",
                    "Unpaved parking lots",
                    "Cars",
                    "Trains",
                    "Stadium seats"]
    ignored_labels = [0]
    return img, gt, rgb_bands, ignored_labels, label_values, palette


def dfc2018_loader(folder):
    palette = None
    img = open_file(folder + '20170218_UH_CASI_S4_NAD83.hdr')[:, :, :-2]
    # '2018_IEEE_GRSS_DFC_GT_TR.hdr'
    # '2018_IEEE_GRSS_DFC_HSI_TR.HDR')
    gt = open_file(folder + '2018_IEEE_GRSS_DFC_GT_TR.tif')
    gt = gt.astype('uint8')

    rgb_bands = (47, 31, 15)

    label_values = ["Unclassified",
                    "Healthy grass",
                    "Stressed grass",
                    "Artificial turf",
                    "Evergreen trees",
                    "Deciduous trees",
                    "Bare earth",
                    "Water",
                    "Residential buildings",
                    "Non-residential buildings",
                    "Roads",
                    "Sidewalks",
                    "Crosswalks",
                    "Major thoroughfares",
                    "Highways",
                    "Railways",
                    "Paved parking lots",
                    "Unpaved parking lots",
                    "Cars",
                    "Trains",
                    "Stadium seats"]
    ignored_labels = [0]
    return img, gt, rgb_bands, ignored_labels, label_values, palette
