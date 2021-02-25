from utils import open_file
import numpy as np

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
