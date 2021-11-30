import imageio
import datetime
import shutil
import glob
import numpy as np
import pandas as pd
import scipy.signal as sg
import scipy.ndimage as ndi

from PIL import ImageFilter, Image
from tslearn.barycenters import \
    euclidean_barycenter
import os
import click
import  open3d as o3d
import time
import fnmatch
import operator
from random import shuffle

import matplotlib.pyplot as plt

#x
#start = 1435
#end = 2015
#start_y =1400
#end_y = 2100

start = 0
start_y=0
end=704
end_y=584


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
    pd.DataFrame.to_csv(df_sum, "sum.csv", index=False)


def find_peak(im):
    peaks, props = sg.find_peaks(im, height=27, width=30)
    max_b = 0
    if len(peaks) > 0:
        max_ind = np.argmax(props["peak_heights"])
        max_b = peaks[max_ind]
    return max_b


#    results_full = sg.peak_widths(img[j], peaks, rel_height=1)


# widest_peak= np.argmax(results_full[0])
# b=img[j][peaks[widest_peak]]
def calculate_mkm(band):
    #!!!!!!!!!!
    if start ==0:
        band +=1435
    else:
        band += start

    # =369+0,484*A4


    #
    ## 22 09 2021 kalib change frpom nm = 369 + 0.484 * band
    mkm = -341.099 + 0.51639*band
    # nm=band
    # у = -2.98 + 0, 0068 * x + (-4.17)e - 6 * x~2
    # mm =float( -2.98 + 0.0068 * nm + (-4.17) * pow(10, -6) * pow(nm, 2))

    # mm formula changed 28 06 2021
    #mm = (-2.22 + 0.0068 * nm + (-4.178) * pow(10, -6) * pow(nm, 2))
    # return pozitive mkm to get integer values
    return (1) * mkm

def calculate_fast_middle_mass(img):
    x=img
    center_of_mass = (x * np.arange(len(x))).sum() / x.sum()
    return center_of_mass

def calculate_middle_mass(img):
    #!!!!!!!!!!!!!
    return 50
#    !!!!!!!!!!!!
    s_sum = 0
    s_delta = 0
    for i in range(0, len(img) - 1):
        s_sum += i * ((img[i] + img[i + 1]) / 2)
        s_delta += (img[i] + img[i + 1]) / 2
    x = s_sum / s_delta
    return x


def sum_lines(img, fname, koef,start_x,stop_x, start_y, stop_y):
    """
    writes summary data to .txt file representing the
    overall brightness of the     each of 1535 dots along all spectra.
    Also writes max reflect  value
    and band with max brightness
    also writes mkm depth values calculated via different methods
    """
    # 100 abs limit
    # img=np.where(img>100,img,0 )

    # band = (nm-369)/0.484
    # 450nm - 167 band
    # 700nm - 683 band
    # use only these 167 -683 bands
    initial_size = img.shape
    #!!!! uncomment for 32 bit
    #img = img[:,:,1]

    #start_band = 166
    #end_band= 684
    start_band = start_x
    end_band = stop_x
#1400 2400initial_size[1]

    #img = np.delete(img, slice(0, start_band, 1), 1)
    #img = np.delete(img, slice(end_band-start_band, -1, 1), 1)
    #img = np.delete(img, slice(0, start_y, 1), 0)
    #img = np.delete(img, slice(stop_y-start_y, -1,1), 0)

    koef = np.delete(koef, slice(0, start_band, 1), 0)
    koef = np.delete(koef, slice(end_band-start_band, -1, 1), 0)
    #koef = np.delete(koef, slice(0, start_y, 1), 1)
    #koef = np.delete(koef, slice(stop_y - start_y, -1, 1),1)

    res = []
    if fname[0] == 't':
        img_transformed = img * np.array(koef)[np.newaxis, :]
    else:
        img_transformed = img

    for j in range(0, img.shape[0]):  # 1536
        res.append((j,calculate_mkm(calculate_fast_middle_mass(img[j]))))
        #!!!!!!!!
        continue

        # A good compromise consists in calculating the barycentre of the peak area, e.g.,
        # the portion above 50 % of the peak intensity
        max_of_line = np.amax(img[j])
        img_50above = np.where(img[j] > max_of_line * 0.5, img[j] - max_of_line * 0.5, 0)
        #np.transpose get_barycenter(img[j])
        cm_scipy_50=ndi.measurements.center_of_mass(img_50above)
        cm_scipy=ndi.measurements.center_of_mass(img[j])

        #used for 2048 images
        img_band_trimmed = np.delete(img[j], slice(0, start_band, 1), 0)
        img_band_trimmed = np.delete(img[j], slice(end_band, -1, 1), 0)

        img_band_trimmed = img[j]

        cm_scipy_50_band_trimmed=ndi.measurements.center_of_mass(img_band_trimmed)

        max_of_line_band_trimmed = np.amax(img_band_trimmed)
        img_band_trimmed_50above = np.where(img_band_trimmed > max_of_line_band_trimmed * 0.5,
                                           img_band_trimmed - max_of_line_band_trimmed * 0.5, 0)
        img_band_trimmed_30above = np.where(img_band_trimmed > max_of_line_band_trimmed * 0.3,
                                            img_band_trimmed - max_of_line_band_trimmed * 0.3, 0)
        img_band_trimmed_70above = np.where(img_band_trimmed > max_of_line_band_trimmed * 0.7,
                                            img_band_trimmed - max_of_line_band_trimmed * 0.7, 0)
        img_band_trimmed_10above = np.where(img_band_trimmed > max_of_line_band_trimmed * 0.1,
                                            img_band_trimmed - max_of_line_band_trimmed * 0.1, 0)
        img_band_trimmed_90above = np.where(img_band_trimmed > max_of_line_band_trimmed * 0.90,
                                            img_band_trimmed - max_of_line_band_trimmed * 0.90, 0)

        cm_scipy_band_trimmed_30above = ndi.measurements.center_of_mass(img_band_trimmed_30above)
        cm_scipy_band_trimmed_50above = ndi.measurements.center_of_mass(img_band_trimmed_50above)
        cm_scipy_band_trimmed_70above = ndi.measurements.center_of_mass(img_band_trimmed_70above)
        cm_scipy_band_trimmed_10above = ndi.measurements.center_of_mass(img_band_trimmed_10above)
        cm_scipy_band_trimmed_90above = ndi.measurements.center_of_mass(img_band_trimmed_90above)

        max_band_scipy = find_peak(img[j])
        max_band_scipy_transformed = find_peak(img_transformed[j])
        #t=calculate_mkm(int(cm_scipy_band_trimmed_50above[0]))
        # if j % 100 == 0:
        # plt.plot(peaks, img[j][peaks], "x")
        # plt.plot(img[j])

        max_band = np.argmax(img[j])
        max_band_transformed = np.argmax(img_transformed[j])

        res.append((j,
                    np.amax(img[j]),
                    np.sum(img_transformed[j]),
                    np.sum(img[j]),
                    np.amax(img_transformed[j]),
                    max_band,
                    max_band_transformed,
                    calculate_mkm(max_band),
                    calculate_mkm(max_band_transformed),
                    calculate_mkm(max_band_scipy),
                    calculate_mkm(max_band_scipy_transformed),
                    calculate_mkm(calculate_middle_mass(img[j])),
                    calculate_mkm(calculate_middle_mass(img_transformed[j])),
                    cm_scipy_band_trimmed_50above[0],
                    calculate_mkm(cm_scipy_band_trimmed_50above[0]),
                    calculate_mkm(calculate_fast_middle_mass(img[j])),
                    calculate_mkm(cm_scipy_band_trimmed_30above[0]),
                    calculate_mkm(cm_scipy_band_trimmed_70above[0]),
                    calculate_mkm(cm_scipy_band_trimmed_10above[0]),
                    calculate_mkm(cm_scipy_band_trimmed_90above[0]),
                    calculate_mkm(cm_scipy_50_band_trimmed[0])
                   ))
    res = np.array(res)
    res = np.uint(res)
    np.savetxt(fname + ".csv", res, fmt='%i', delimiter=',',
               header="x,mkm_fast_middle_mass", comments=''
               )
    #!!!!!!!!
    return 555
    np.savetxt(fname + ".csv", res, fmt='%i', delimiter=',',
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
                      "mkm_mass_c_transformed,"
                      "cm_scipy_band_trimmed_50above,"
                      "mkm_scipy_band_trimmed_50above,"
                      "mkm_fast_middle_mass,"
                      "mkm_scipy30,"
                      "mkm_scipy70,"
                      "mkm_scipy10,"
                      "mkm_scipy90,"
                      "mkm_scipy_all"
    , comments=''

               )

    #plt.show()


def avg_spectra(fname):
    """
    writes summary data to .txt file representing the overall brightness of the
    each of 1535 dots along all spectra. Also writes max value
    """

    im = Image.open(fname).convert('L')
    im = im.crop((start, start_y, end, end_y))
    img=np.array(im)
    avg_all = int(np.average(img))
    res = []
    for j in range(0, img.shape[1]):  # 1536
        avg = np.average(img[:, j])
        res.append((j,
                    avg,
                    float(avg_all / avg)
                    # abs max np.sum(img[j]) / 200

                    ))
    res = np.array(res)

    np.savetxt( fname + "avg" +".txt", res, fmt='%f', delimiter='\t',
               header="band\tbrightness\tkoef", comments='')
    return res[:, 2]


filters = [ImageFilter.GaussianBlur,
           ImageFilter.ModeFilter,
           ImageFilter.MedianFilter,
           ImageFilter.UnsharpMask,
           ImageFilter.BoxBlur(5),
           ImageFilter.BLUR,
           ImageFilter.CONTOUR,
           ImageFilter.DETAIL,
           ImageFilter.EDGE_ENHANCE,
           ImageFilter.EDGE_ENHANCE_MORE,
           ImageFilter.EMBOSS,
           ImageFilter.FIND_EDGES,
           ImageFilter.SHARPEN,
           ImageFilter.SMOOTH,
           ImageFilter.SMOOTH_MORE,
           ImageFilter.SMOOTH]
filters = [ImageFilter.BoxBlur,
           ImageFilter.GaussianBlur,
           ImageFilter.SMOOTH_MORE,
           ImageFilter.SMOOTH
           ]


def get_barycenter(img):
    from tslearn.barycenters import \
        euclidean_barycenter, \
        dtw_barycenter_averaging, \
        dtw_barycenter_averaging_subgradient, \
        softdtw_barycenter

    arr=[]
    for i in img:
        arr.append([i])
    bar = euclidean_barycenter(arr)
    return bar


def bc(im, coord):
    # dtw_barycenter_averaging, \
    # dtw_barycenter_averaging_subgradient, \
    # softdtw_barycenter
    from tslearn.datasets import CachedDatasets

    # fetch the example data set
    # numpy.random.seed(0)
    # X_train, y_train, _, _ = CachedDatasets().load_dataset("Trace")
    # X = X_train[y_train == 2]
    X = im
    length_of_sequence = X.shape[1]
    ax1 = plt.subplot(10, 1, coord / 10 + 1)
    ax1.set_xlim([0, length_of_sequence])

    def plot_helper(barycenter):
        # plot all points of the data set
        cntr = 0
        for series in X:
            if (cntr % 100 == 0):
                plt.plot(series.ravel(), "k-", alpha=.2, )
                plt.legend()
            cntr += 1
        # plot the given barycenter of them
        plt.plot(barycenter.ravel(), "r-", linewidth=2)

    # plot the four variants with the same number of iterations and a tolerance of
    # 1e-3 where applicable

    plt.title("Euclidean barycenter at line coord=" + str(coord))
    plot_helper(euclidean_barycenter(im))

    # plt.subplot(4, 1, 2, sharex=ax1)
    # plt.title("DBA (vectorized version of Petitjean's EM)")
    # plot_helper(dtw_barycenter_averaging(im, max_iter=50, tol=1e-3))

    # plt.subplot(4, 1, 3, sharex=ax1)
    # plt.title("DBA (subgradient descent approach)")
    # plot_helper(dtw_barycenter_averaging_subgradient(im, max_iter=50, tol=1e-3))
    #
    # plt.subplot(4, 1, 4, sharex=ax1)
    # plt.title("Soft-DTW barycenter ($\gamma$=1.0)")
    # plot_helper(softdtw_barycenter(im, gamma=1., max_iter=50, tol=1e-3))

    # clip the axes for better readability

    # show the plot(s)


def apply_filters(fname):
    a = np.uint8(imageio.imread(fname))
    # im = Image.open(fname, "L")
    im = Image.fromarray(a, "L")
    a2 = np.array(im)

    for f in filters:
        if f.name == 'BoxBlur' or f.name == 'GaussianBlur':
            for i in range(15):
                filtered = np.array(im.filter(f(radius=i)))
                imageio.imwrite(uri="out/" + fname + "_filter_" + f.name + "_" + str(i) + ".tif", im=filtered,
                                format="tiff")
                # filtered.save("out/"+nm+"_filter_"+f.name+"_max2d.tif", format="tiff")
                np.savetxt("out/" + fname + "_filter_" + f.name + "_" + str(i) + ".txt", filtered, fmt='%i',
                           delimiter=',',
                           comments='')

        else:
            filtered = np.array(im.filter(f))
            imageio.imwrite(uri="out/" + fname + "_filter_" + f.name + ".tif", im=filtered, format="tiff")
            # filtered.save("out/"+nm+"_filter_"+f.name+"_max2d.tif", format="tiff")
            np.savetxt("out/" + fname + "_filter_" + f.name + ".txt", filtered, fmt='%i', delimiter=',', comments='')


def pl3d():
    import os
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import numpy as np

    # fig = plt.figure()
    # ax = Axes3D(fig)
    #
    # my_data = np.genfromtxt('out/5max2d.tif_filter_BoxBlur_0.txt_3col.txt', delimiter=',', skip_header=0)
    # my_data[my_data == 0] = 1
    # my_data = my_data[~np.isnan(my_data).any(axis=1)]
    # X = my_data[:, 0]
    # Y = my_data[:, 1]
    # Z = my_data[:, 2]
    # surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
    #                        linewidth=0, antialiased=False, rcount=200, ccount=200)
    #
    # # Customize the z axis.
    # ax.set_zlim(-1.01, 1.01)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    #
    # # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.show()
    #
    #
    # xi = np.linspace(X.min(), X.max(), int(len(Z) / 3))
    # yi = np.linspace(Y.min(), Y.max(), int(len(Z) / 3))
    # zi = griddata((X, Y), Z, (xi[None, :], yi[:, None]), method='nearest')
    #
    # xig, yig = np.meshgrid(xi, yi)
    #
    # surf = ax.plot_surface(xig, yig, zi, cmap='gist_earth')
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # ax.set_title('2014 ATM Data 0.01 Degree Spacing')
    # ax.set_xlabel('Latitude')
    # ax.set_ylabel('Longitude')
    # ax.set_zlabel('Elevation (m)')
    # ax.set_zlim3d(auto=True)
    #
    # # This import registers the 3D projection, but is otherwise unused.

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    X = np.arange(-5, 5, 0.05)
    Y = np.arange(-5, 5, 0.05)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X ** 2 + Y ** 2)
    Z = np.sin(R)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False, rcount=200, ccount=200)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)


def get_max_tif():
    """Geneates max tiff based on the tif series filename and max pixel level .
        Basically creates depth map
    """
    max_array = []
    for filepath in glob.iglob('metal\*.tif'):  # 0?
        img = imageio.imread(filepath)
        nm = filepath[6]
        mkm = int(str(filepath[-6] + filepath[-5]))
        print(mkm)
        if mkm == 0:
            a = np.empty((len(img), len(img[0])), int)
            a.fill(0)
            # a = [[0 for i in range(len(img))] for j in range(len(img[0]))]
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
    imageio.imwrite(uri=nm + "max2d.tif", im=a, format="tiff")
    im = Image.fromarray(a, "L")
    im.save("out/" + nm + "_max2d.tif", format="tiff", )
    print(nm + "max2d.tif saved.")


def get_3col_txt_from_txt(filepath, x_c, y_c, z_c):

    a_out = []
    a_in = np.transpose(np.genfromtxt(filepath, delimiter=',', filling_values=np.nan, case_sensitive=True,
                         deletechars='',
                         replace_space=' ', skip_header=0)
                        )
    if type(a_in[0]) == np.float64 :
        #one pixel files
        length = 1
        for i in range(0, len(a_in)):
            a_out.append([i * x_c, 0, 1000 - (a_in[i] * z_c)])
    else:
        length = len(a_in[0])
        for i in range(0, len(a_in)):
            for j in range(0, (length)):
                # Width: 921.3630
                # µm(512)
                # Height: 921.3630
                # µm(512)
                # Depth: 118
                # µm(59)

                a_out.append([i * x_c, j * y_c,1000- (a_in[i, j] * z_c)])
    s=str.replace(str(x_c)+"-"+str(y_c)+"-"+str(z_c),".",",")
    np.savetxt(filepath.replace("2d.txt", "")
               +"_"+s+ "_3col.csv"
    , a_out, fmt='%.1f', delimiter=' ', comments='')
    print(x_c,y_c,z_c)


def get_tif_from_csv(path,suffix):
    """
    Reads csv in batch and creates  tiffs based on column number in csv
    :return:
    """
    print(path)
    if  os.path.exists(path+suffix+"out"):
        shutil.rmtree(path+suffix+"out", ignore_errors=True)
    os.makedirs(path +suffix+ "out")
    line = np.recfromcsv(path+'/'+[s for s in os.listdir(path) if s.endswith('.tif.csv')][0], delimiter=',', filling_values=np.nan, case_sensitive=True, deletechars='',
                         replace_space=' ', names=True)
    print(line.dtype.names)
    for index, col in enumerate(line.dtype.names):
        print(col,"-----")
        img = []
        for filepath in glob.iglob(path+'\*.tif.csv'):
            fname=os.path.basename(filepath)
            #j = int(str(filepath[5] + filepath[6]))
            line = np.genfromtxt(filepath, delimiter=',', filling_values=np.nan, case_sensitive=True,
                                 deletechars='',
                                 replace_space=' ', skip_header=1)
            #print(fname)
            img.append(np.uint((line[:, index])))
        #formats with error
        #im = Image.fromarray(np.array(img), "L")
        #im.save(path+suffix+"out/" + col + "2d" + ".tif", format="tiff", )
        imageio.imwrite(uri=path+suffix+"out/" + col + "2d" + ".tif", im=np.array(img), format="tiff", )
        np.savetxt(path+suffix+"out/" + col + "2d" + ".txt", img, fmt='%i', delimiter=',', comments='')
        get_3col_txt_from_txt(path + suffix + "out/" + col + "2d" + ".txt", 1.4, 5, 1)
	#a=[1]#,2,5,10,25]
        #for x in a:
         #   for y in a:
          #      get_3col_txt_from_txt(path + suffix + "out/" + col + "2d" + ".txt", x, y, 1)


def find_4max(fname):
    """Finds 4 max for 4 lines for each line"""
    lines=[[1420,1460],
           [1490,1520],
           [1700,1740],
           [1760,1840]]
    start_y=1230
    stop_y=1930
    from PIL import Image

    # Opens a image in RGB mode
    im = Image.open(fname).convert('L')
    # Size of the image in pixels (size of original image)
    # (This is not mandatory)
    width, height = im.size
    # Setting the points for cropped image
    left = 5
    top = height / 4
    right = 164
    bottom = 3 * height / 4
    # Cropped image of above dimension
    # (It will not change original image)
    im1 = im.crop((0, 1230, width, 1930))
    # Shows the image in image viewer
    #im1.show()
    array=[]
    ln=[]
    img = np.asarray(im1)
    # !!!! uncomment for 32 bit
#    img = img1[:, :, 1]
    #np.savetxt("k4all" + ".csv", img, fmt='%i', delimiter=',')
    #img = np.delete(img, slice(0, start_y, 1), 1)
    #img = np.delete(img, slice(stop_y - start_y, -1, 1), 1)
    #img = np.transpose(img)
    #np.savetxt("k4" + ".csv", img, fmt='%i', delimiter=',')
    for i in range(0,(img.shape[0]-1)):
        line=img[i]
        #print(i)
        ln = []
        for j in range(0,4):
            max = np.argmax(line[lines[j][0]:lines[j][1]])+lines[j][0]
            ln.append(max)
        array.append(ln)

    #np.savetxt("k4_max" + ".csv", array, fmt='%i', delimiter=',',header="l1,l2,l3,l4")
    for j in range(0, 4):
        print(calculate_mkm(ln[j]))
    return array
def transform(filepath,imag,led):
    #res=[]
    res=np.uint(np.divide(imag,led))
    for i in range (1,255):
        out = res*i
        imageio.imwrite(uri=filepath + "_divided"+str(i)+".tiff", im=np.array(out), format="tiff", )
        print(i)
    imageio.imwrite(uri=filepath +"_divided.tiff", im=np.array(res), format="tiff", )

def split_dir(dir,lines):
    """split dir based on lines number . 2500 tiffs and 500 lines = 5 dirs X 500"""

    return (copy_files(os.path.abspath(dir),lines))


  # the number of files in seach subfolder folder
"""
Create sutract files 
"""
def create_diff_files(dir):


    dirs=[]
    for f in glob.iglob(dir + '\\00*'):
        if os.path.isdir(f):
            dirs.append(f)
    for i in range(0,len(dirs)-1):
        dir1=dirs[i]
        dir2=dirs[i+1]
        outdir = dir + '/subt' + os.path.basename(dir1)[0:5] + "-" + os.path.basename(dir2)[0:5]
        if os.path.exists(outdir):
            shutil.rmtree(outdir, ignore_errors=True)
        os.mkdir(outdir)
        print("Subtracting dir ", dir2, "from", dir1)
        for  filepath in glob.iglob(dir1+'\*.tif'):
            fname = os.path.basename(filepath)
            im = Image.open(filepath)#.convert('L')
            img = np.asarray(im)
            img1 = img.astype(np.int32)

            im = Image.open(dir2+"/"+fname)#.convert('L')
            # im = im.crop((start, start_y, end, end_y))
            img = np.asarray(im)
            img2 = img.astype(np.int32)

            img = np.subtract(img1, img2)
            img[img < 0] = 0



            imageio.imwrite(uri=outdir+"/" + fname, im=np.array(img), format="tiff", )
            np.savetxt(outdir+"/" + fname + ".txt", img, fmt='%i', delimiter=',', comments='')
            get_3col_txt_from_txt(outdir+"/" + fname + ".txt", 1.4, 5, 1)


def copy_files(abs_dirname,N):
    """copy files into subdirectories."""


    file_list=[]
    files = [os.path.join(abs_dirname, f) for f in os.listdir(abs_dirname)]

    i = 0
    curr_subdir = None
    f=glob.iglob(abs_dirname + '\*.tif')
    out=[]
    for f in glob.iglob(abs_dirname + '\*.tif'):
        # create new subdir if necessary
        if i % N == 0:
            subdir_name = os.path.join(abs_dirname, str(int(i / N + 1)).zfill(5))
            print("Copying files to",subdir_name)
            out.append(subdir_name)
            if os.path.exists(subdir_name):
                shutil.rmtree(subdir_name, ignore_errors=True)
            os.mkdir(subdir_name)
            curr_subdir = subdir_name

        # move file to current dir
        f_base = os.path.basename(f)

        shutil.copy(f, os.path.join(subdir_name, f_base))
        file_list.append([str(int(i / N + 1)).zfill(5),f_base])
        i += 1
    np.savetxt(abs_dirname+"/filelist.txt", file_list, fmt='%s', delimiter=',', comments='')

    return out

def generate_mesh(fname):
    pcd = o3d.io.read_point_cloud(fname, 'xyz')
    poisson_mesh =     o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]
    o3d.io.write_triangle_mesh("bpa_mesh.ply", poisson_mesh)

@click.command()
@click.pass_context
@click.option('--dir', help='Directory with tiffs, MIN- prefix for finding newest ', required=True, metavar='PATH')
@click.option('--lines', help='Number of lines in each dir ', required=False,type=int )
@click.option('--get_only_tiff', help='set to 0 to make csv, 1- process existing csv, 555- show final tiff ', required=False,type=int )
@click.option('--final', help='Number of files after which final 3d pic should be displayed  ', required=False,type=int )


def func(
    ctx: click.Context,
    dir: str,
    lines: int,
    final: int,
    get_only_tiff: int
):

    #path= "2021-09-17-10-37-39.0511242-1"
    #path = "2021-09-17-10-37-39.0511242"
		
    #path= "2021-09-30-10-56-10.3523425" #1816
    #path="2021-09-30-10-40-12.4487604" #829
    #для того архива, где меньше изображений вычитай и скаждой тифки шум 302 -829
     #для другого используй шум 302 без кубика 1816
    #path="2021-10-06-14-37-59.8220891_800"
    #path="2021-10-06-15-38-43.5490766_500"

    fname="mkm_fast_middle_mass_1,4-5-1_3col.csv"

    if dir[0:4] == "MIN-":
        currentDateTime = datetime.datetime.now()
        date = currentDateTime.date()
        year = date.strftime("%Y")
        set1 = set(glob.glob(os.path.join(dir[4:], year+'*/'))) -set(glob.glob(os.path.join(dir[4:], year+'*out/')))
        dir=  max( set1,                                                     key=os.path.getmtime)
        print( "The newest directory is", dir)
    while True and get_only_tiff != 555:
        if lines !=0:
            dirs = split_dir(dir,lines)

            for d in dirs:
                make_tifs(d, get_only_tiff)
                show3d(d + "_X" + str(start) + "_" + str(end) + "-Y" + str(start_y) + "_" + str(
                    end_y) + "out/mkm_scipy70_1,4-5-1_3col.csv")

                if os.path.exists(d):
                    print("Deleting ",d)
                    shutil.rmtree(d, ignore_errors=True)
        else:
            make_tifs(dir,get_only_tiff)
            file_num = len(fnmatch.filter(os.listdir(dir), '*.tif'))
            show3d(dir + "_X" + str(start) + "_" + str(end) + "-Y" + str(start_y) + "_" + str(
                end_y) + "out/"+fname,False,file_num)

            if file_num > final:
                break
            #create_diff_files(dir)
    file_num = final

    show3d(dir + "_X" + str(start) + "_" + str(end) + "-Y" + str(start_y) + "_" + str(
            end_y) + "out/"+"mkm_fast_middle_mass_1,4-5-1_3col.csv",True,file_num)
          #  "C:/Users\LRS\PycharmProjects\HSI_depth/2021-10-06-15-38-43.5490766_500/00001_X0_704-Y0_584out\mkm_scipy70_1,4-5-1_3col.csv")


def make_tifs(dir, get_only_tif):
    print(datetime.datetime.now().time())
    noise_path = 'calib/шум 302 без кубика.tif'
    #noise_path = 'calib/шум 302.tif'
    path = dir
    #set to 1 if you ned to get only out tif without recalculating csvs
    get_only_tif = get_only_tif


    if get_only_tif!=1:
        res = []
        # for i in range (0,1000):
        #     res.append([i,calculate_mkm(i)])
        koef = avg_spectra("calib\спектр.tif")
        cnt = 0
        led = np.asarray(Image.open("calib\спектр.tif").convert('L'))

        for item in os.listdir(path):
            if item.endswith(".csv"):
                os.remove(os.path.join(path, item))
                pass

        print("Total ",len(list(glob.iglob(path+'\*.tif'))))
        for filepath in glob.iglob(path+'\*.tif'):
            if filepath == "led.tif": continue
            im = Image.open(filepath).convert('L')
            im = im.crop((start, start_y, end, end_y))
            img = np.asarray(im)
            img = img.astype(np.int16)

            noise = Image.open(noise_path).convert('L')
            ns=np.asarray(noise)
            ns= ns.astype(np.int16)

            #Sutract noise
            #img = np.subtract(img,ns)
            #img[img <0 ] = 0

            print(filepath)

            if (cnt % 10 == 0):
                pass
                #bc(np.transpose(img), cnt)

            cnt += 1
            sum_lines(img, filepath, koef, start, end,start_y, end_y)

    get_tif_from_csv(path,"_X"+str(start)+"_"+str(end)+"-Y"+str(start_y)+"_"+str(end_y))
    print(datetime.datetime.now().time())






def show3d(fname,final,num):

    from mpl_toolkits import mplot3d
    #generate_mesh(fname)
    # actual code to load,slice and display the point cloud
    #fname= "sample_w_normals.xyz"
    outdir = fname.split("out/")[0]+"out/"
    point_cloud = np.loadtxt(fname, skiprows=1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    arr = []
    arr = np.array([[155, 155 ,155] for i in range(point_cloud.shape[0])])
    #print(arr)
   # pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:6] / 255)
    pcd.colors = o3d.utility.Vector3dVector(arr[:,:3] / 255)

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    #pcd.normals = o3d.utility.Vector3dVector(point_cloud[:, 6:9])
    poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]
    o3d.io.write_triangle_mesh(outdir+"poisson_mesh.ply", poisson_mesh)

    cloud = o3d.io.read_point_cloud(fname, 'xyz')  # Read the point cloud
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name= str(len(cloud.points))+ " points in "+ str(num)+ " files.",
                      width=1000,height=1000)
    vis.add_geometry(cloud)
    vis.poll_events()
    vis.update_renderer()
    time.sleep(3)
    #vis.add_geometry(cloud)
    #cloud = o3d.io.read_image(fname)
    #o3d.visualization.draw_geometries([cloud])
    #o3d.visualization.draw_geometries_with_custom_animation([cloud])
    #

    #vis.destroy_window()
    #open3d.geometry.draw_geometries([cloud])  # Visualize the point cloud
    if final == True:
        vis.destroy_window()
        o3d.visualization.draw_geometries( [cloud], window_name = "Finally "+str(len(cloud.points))+ " points in "+ str(num)+ " files" )
        file_data_path = fname  # "N.xyz"
        # file_data_path = "sample.xyz"

        point_cloud = np.loadtxt(file_data_path)
        print("Read " + str(point_cloud.shape[0]) + " points from " + fname)
        shuffle(point_cloud)
        #point_cloud = point_cloud[:50000]
        print("Left  " + str(point_cloud.shape[0]))
        #mean_Z = np.mean(point_cloud, axis=0)[2]
        spatial_query = point_cloud  # [abs(point_cloud[:, 2] - mean_Z) < 1]
        xyz = spatial_query[:, :3]
        # rgb = spatial_query[:, 3:]
        ax = plt.axes(projection='3d',
                      title = str( point_cloud.shape[0]) + " points in " + fname,
                       )
        ax.set_xlabel("Координата Х, мкм")
        ax.set_ylabel("Координата Y, мкм")
        ax.set_zlabel("Координата Z, мкм")
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=0.01)  # rgb / 255
        plt.gcf().set_size_inches((20, 20))
        plt.show()


if __name__ == '__main__':
    #show3d(
    #    'C:/Users\LRS\PycharmProjects\HSI_depth/2021-10-06-15-38-43.5490766_500/00001_X0_704-Y0_584out\mkm_scipy702d.tif') mkm_scipy70_1,4-5-1_3col.csv
    func()
