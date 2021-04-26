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
#import statsmodels.api as sm
import seaborn as sns
import pylab as py
warnings.filterwarnings('ignore')


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
        'img': '3.tif',
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
    palette = None
    img = open_file(folder + CUSTOM_DATASETS_CONFIG['Depth']['img'])  # [:, :, :-2]
    calibr = np.mean(open_file(folder + CUSTOM_DATASETS_CONFIG['Depth']['calibr']), axis = 0 )

    # removal of damaged sensor line
    img[1122] = img[1122,0]
    #img = np.delete(img, 1122, 0)
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

    print("File " ,CUSTOM_DATASETS_CONFIG['Depth']['img'] , "has max value=",max,"located at ",  max_index   )
    img = img.astype('uint8')
    guess_distribution(img)
    time.sleep(5)
    np.savetxt(CUSTOM_DATASETS_CONFIG['Depth']['img']+".txt",img,fmt='%i', delimiter=',' )
    print("Line Position MaxValue")
    indices = list( np.where((img > CUSTOM_DATASETS_CONFIG['Depth']['limit']).any(axis=1)))
    it = np.nditer(indices, flags = ['f_index'])
    for i in  it :
        row = img[i]
        max_index = np.argmax(row)
        print(i," ", max_index, " ", row[max_index])
       # val = img[i]#[index]

    #indices = indices[4:]
    #indices = np.delete(indices,
    #indices = np.delete( indices,
    #img = img[:, :, get_good_indices()]  # [:, :, get_good_indices(name)]

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
            gt[i,j] = 1
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

def guess_distribution(img):
    df = pd.DataFrame(img)
    #print(df.head())
    print(df.columns)

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
