import imageio
import glob
import numpy as np
import pandas as pd
import scipy.signal as sg
import matplotlib.pyplot as plt


# import find_peaks

def combine_files():
    """
    Coombines sum files into single csv
    :return:
    """
    df_sum = pd.DataFrame()
    i = 0
    columns = ['x']
    for filepath in glob.iglob('sum*.txt'):
        columns.append(filepath[3:9])
        columns.append("t_" + filepath[3:9])
        print(filepath)
        df = pd.read_csv(filepath, header=None)
        if i == 0:
            df_sum[0] = df[0]
        df_sum = pd.concat([df_sum, df[2]], axis=1)
        df_sum = pd.concat([df_sum, df[3]], axis=1)
        i += 1
    df_sum.columns = columns
    pd.DataFrame.to_csv(df_sum, "sum.csv", index=None)


def find_peak(im):
    peaks, props = sg.find_peaks(im, height=22, width=30)
    max_b = 0
    if len(peaks) > 0:
        max_ind = np.argmax(props["peak_heights"])
        max_b = peaks[max_ind]
    return max_b


#    results_full = sg.peak_widths(img[j], peaks, rel_height=1)


# widest_peak= np.argmax(results_full[0])
# b=img[j][peaks[widest_peak]]
def calculate_mkm(band):
    # =369+0,484*A4
    nm = 369 + 0.484 * band
    # nm=band
    # у = -2.98 + 0, 0068 * x + (-4.17)e - 6 * x~2
    mm = -2.98 + 0.0068 * nm + (-4.17) * pow(10, -6) * pow(nm, 2)
    # return mkm to get integer values
    return (-1) * mm * 1000


def calculate_middle_mass(img):
    s_sum = 0
    s_delta = 0
    for i in range(0, len(img) - 1):
        s_sum += i * ((img[i] + img[i + 1]) / 2)
        s_delta += (img[i] + img[i + 1]) / 2
    x = s_sum / s_delta
    return x


def sum_lines(img, fname, koef):
    """
    writes summary data to .txt file representing the
    overall brightness of the     each of 1535 dots along all spectra.
    Also writes max reflect  value
    and band with max brightness
    """

    res = []
    if fname[0] == 't':
        img_transformed = img * np.array(koef)[np.newaxis, :]
    else:
        img_transformed = img
    for j in range(0, 1535):  # 1536
        max_band_scipy = find_peak(img[j])
        max_band_scipy_transformed = find_peak(img_transformed[j])

        # if j % 100 == 0:
        # plt.plot(peaks, img[j][peaks], "x")
        # plt.plot(img[j])

        max_band = np.argmax(img[j])
        max_band_tranformed = np.argmax(img_transformed[j])

        res.append((j,
                    np.amax(img[j]),
                    np.sum(img_transformed[j]),
                    np.sum(img[j]),
                    np.amax(img_transformed[j]),
                    max_band,
                    max_band_tranformed,
                    calculate_mkm(max_band),
                    calculate_mkm(max_band_tranformed),
                    calculate_mkm(max_band_scipy),
                    calculate_mkm(max_band_scipy_transformed),
                    calculate_mkm(calculate_middle_mass(img[j])),
                    calculate_mkm(calculate_middle_mass(img_transformed[j]))
                    ))
    res = np.array(res)
    np.savetxt( fname + ".csv", res, fmt='%i', delimiter=',',
               header="x,"
                      "max,"
                      "sum_transformed,"
                      "sum,"
                      "max_transformed,"
                      "band_max,"
                      "band_max_transformed,"
                      "mkm,"
                      "mkm_transformed,"
                      "mkm_scipy,"
                      "mkm_scipy_transformed,"
                      "mkm_mass_c,"
                      "mkm_mass_c_transformed"
               , comments=''

               )

    plt.show()


def avg_spectra(fname):
    """
    writes summary data to .txt file representing the overall brightness of the
    each of 1535 dots along all spectra. Also writes max value
    """

    img = imageio.imread(fname)
    avg_all = int(np.average(img))
    res = []
    for j in range(0, 2048):  # 1536
        avg = np.average(img[:, j])
        res.append((j,
                    avg,
                    float(avg_all / avg)
                    # abs max np.sum(img[j]) / 200

                    ))
    res = np.array(res)

    np.savetxt("avg" + fname + ".txt", res, fmt='%f', delimiter='\t',
               header="band\tbrightness\tkoef", comments='')
    return res[:, 2]

def get_max_tif():
    """Geneates max tiff based on the tif series filename and max pixel level .
        Basically creates depth map
    """
    max_array = []
    for filepath in glob.iglob('metal\*.tif'):
        img = imageio.imread(filepath)
        mkm = int(str(filepath[-6] + filepath[-5]))
        print(mkm)
        if mkm == 0:
            a = np.empty((len(img), len(img[0])),int)
            a.fill(0)
            #a = [[0 for i in range(len(img))] for j in range(len(img[0]))]
            max_array = img
        for i in range(0, len(img)):
            for j in range(0, len(img[0])):
                if img[i, j] > max_array[i, j]:
                    max_array[i, j] = img[i, j]
                    a[i, j] = mkm

        # a.append(img)
        # i+=1
        # if i==6:
        #     max = np.maximum.reduce([a[0], a[1], a[2] ,a[3], a[4]])
        #     max_array.append(max)
        #     i=0

    # exit
    imageio.imwrite(uri="max.tif",im=a,format="tiff")
    np.savetxt("max.txt", a, fmt='%i', delimiter=',',comments='')
    print("Max.tif saved.")

def get_tif_from_csv():
    """
    Reads csv in batch and creates  tiffs based on column number in csv
    :return:
    """
    line = np.recfromcsv('tif/001.tif.csv', delimiter=',', filling_values=np.nan, case_sensitive=True, deletechars='',
                         replace_space=' ', names=True)
    print(line.dtype.names)
    for index,col in enumerate(line.dtype.names):
        print(col)
        img = []
        for filepath in glob.iglob('tif\*.tif.csv'):
            j = int(str(filepath[5] + filepath[6]))
            line = np.genfromtxt(filepath, delimiter=',', filling_values=np.nan, case_sensitive=True,
                                 deletechars='',
                                 replace_space=' ',skip_header=1)
            print(j)
            img.append(line[:,index])
        imageio.imwrite(uri=col+".tif", im=img, format="tiff")
        np.savetxt(col+".txt", img, fmt='%i', delimiter=',', comments='')


#get_max_tif()
#img = imageio.imread("1max.tif")
get_tif_from_csv()
exit
koef = avg_spectra("led.tif")
for filepath in glob.iglob('tif\*.tif'):
    if filepath == "led.tif": continue
    img = imageio.imread(filepath)
    print(filepath)
    sum_lines(img, filepath, koef)
# combine_files()
