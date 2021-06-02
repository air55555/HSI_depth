import imageio
import glob
import numpy as np
import pandas as pd

def combine_files():
    """
    Coombines sum files into single csv
    :return:
    """
    df_sum = pd.DataFrame()
    i=0
    columns=['x']
    for filepath in glob.iglob('sum*.txt'):
        columns.append(filepath[3:9])
        columns.append("t_"+filepath[3:9])
        print(filepath)
        df=pd.read_csv(filepath,header=None)
        if i == 0:
            df_sum[0]=df[0]
        df_sum=pd.concat([df_sum,df[2]],axis=1 )
        df_sum = pd.concat([df_sum, df[3]], axis=1)
        i+=1
    df_sum.columns=columns
    pd.DataFrame.to_csv(df_sum,"sum.csv",index=None)

def calculate_mm(band):
    # =369+0,484*A4
    nm = 369 + 0.484 * band
    #nm=band
    #у = -2.98 + 0, 0068 * x + (-4.17)e - 6 * x~2
    mm= -2.98 + 0.0068*nm + (-4.17)*pow(10,-6)*pow(nm,2)
    return (-1)*mm*1000

def sum_lines(img,fname,koef):
    """
    writes summary data to .txt file representing the
    overall brightness of the     each of 1535 dots along all spectra.
    Also writes max reflect  value
    and band with max brightness
    """

    res = []
    if fname[0] == 't':
        img_transformed= img * np.array(koef) [np.newaxis,: ]
    else:
        img_transformed = img
    for j in range(0,1535 ):  # 1536

        max_band = np.argmax(img[j])
        max_band_tranformed = np.argmax(img_transformed[j])

        res.append((j,
                    np.amax(img[j]),
                    np.sum(img_transformed[j]),
                    np.sum(img[j]),
                    np.amax(img_transformed[j]),
                    max_band,
                    max_band_tranformed,
                    calculate_mm(max_band),
                    calculate_mm(max_band_tranformed)
                    ))
    res = np.array(res)
    np.savetxt( "sum" + fname + ".txt", res, fmt='%i', delimiter=',',
                header = "x,"
                         "max,"
                         "sum_transformed,"
                         "sum,"
                         "max_transformed,"
                         "band_max,"
                         "band_max_transformed,"
                         "mm,"
                         "mm_transformed,", comments = ''

    )



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
                    float(avg_all/avg)
                      # abs max np.sum(img[j]) / 200

                    ))
    res = np.array(res)

    np.savetxt( "avg" + fname + ".txt", res, fmt='%f', delimiter='\t',
                header = "band\tbrightness\tkoef",comments='')
    return res[:,2]

koef=avg_spectra("led.tif")
for filepath in glob.iglob('*.tif'):
    if filepath == "led.tif": continue
    img = imageio.imread(filepath)
    sum_lines(img,filepath,koef)
combine_files()
