import imageio
import glob
import numpy as np
import pandas as pd
import scipy.signal as sg
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from PIL import ImageFilter, Image
from tslearn.barycenters import \
    euclidean_barycenter


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
    #band += 172
    # =369+0,484*A4
    nm = 369 + 0.484 * band

    # nm=band
    # у = -2.98 + 0, 0068 * x + (-4.17)e - 6 * x~2
    # mm =float( -2.98 + 0.0068 * nm + (-4.17) * pow(10, -6) * pow(nm, 2))

    # mm formula changed 28 06 2021
    mm = (-2.22 + 0.0068 * nm + (-4.178) * pow(10, -6) * pow(nm, 2))
    # return pozitive mkm to get integer values
    return (1) * mm * 1000


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
    also writes mkm depth values calculated via different methods
    """
    # 100 abs limit
    # img=np.where(img>100,img,0 )

    # band = (nm-369)/0.484
    # 450nm - 167 band
    # 700nm - 683 band
    # use only these bands
    img = np.delete(img, slice(0, 166, 1), 1)
    img = np.delete(img, slice(684, -1, 1), 1)
    koef = np.delete(koef, slice(0, 166, 1), 0)
    koef = np.delete(koef, slice(684, -1, 1), 0)
    res = []
    if fname[0] == 't':
        img_transformed = img * np.array(koef)[np.newaxis, :]
    else:
        img_transformed = img

    for j in range(0, 1535):  # 1536
        # A good compromise consists in calculating the barycentre of the peak area, e.g.,
        # the portion above 50 % of the peak intensity
        max_of_line = np.amax(img[j])
        img_50above = np.where(img[j] > max_of_line * 0.5, img[j] - max_of_line * 0.5, 0)
        #np.transpose get_barycenter(img[j])
        cm_scipy_50=ndi.measurements.center_of_mass(img_50above)
        cm_scipy=ndi.measurements.center_of_mass(img[j])
        img_band_trimmed = np.delete(img[j], slice(0, 166, 1), 0)
        img_band_trimmed = np.delete(img[j], slice(684, -1, 1), 0)
        cm_scipy_50_band_trimmed=ndi.measurements.center_of_mass(img_band_trimmed)
        max_of_line_band_trimmed = np.amax(img_band_trimmed)
        img_band_trimmed_50above= np.where(img_band_trimmed > max_of_line_band_trimmed * 0.5, img_band_trimmed - max_of_line_band_trimmed * 0.5, 0)
        cm_scipy_band_trimmed_50above = ndi.measurements.center_of_mass(img_band_trimmed_50above)
        max_band_scipy = find_peak(img[j])
        max_band_scipy_transformed = find_peak(img_transformed[j])
        t=calculate_mkm(int(cm_scipy_band_trimmed_50above[0]))
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
                    calculate_mkm(cm_scipy_band_trimmed_50above[0])
                    ))
    res = np.array(res)
    res = np.uint(res)
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
                      "mkm_scipy_band_trimmed_50above"
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
    bar = euclidean_barycenter(img)
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
    a_in = np.genfromtxt(filepath, delimiter=',', filling_values=np.nan, case_sensitive=True,
                         deletechars='',
                         replace_space=' ', skip_header=1)
    for i in range(0, len(a_in)):
        for j in range(0, len(a_in[0])):
            # Width: 921.3630
            # µm(512)
            # Height: 921.3630
            # µm(512)
            # Depth: 118
            # µm(59)

            a_out.append([i * x_c, j * y_c, (a_in[i, j] * z_c)])

    np.savetxt(filepath.replace("2d.txt", "") + "_3col.txt", a_out, fmt='%.1f', delimiter=',', comments='')
    print(filepath + "_3col.txt saved.")


def get_tif_from_csv():
    """
    Reads csv in batch and creates  tiffs based on column number in csv
    :return:
    """
    line = np.recfromcsv('tif/001.tif.csv', delimiter=',', filling_values=np.nan, case_sensitive=True, deletechars='',
                         replace_space=' ', names=True)
    print(line.dtype.names)
    for index, col in enumerate(line.dtype.names):
        print(col)
        img = []
        for filepath in glob.iglob('tif\*.tif.csv'):
            j = int(str(filepath[5] + filepath[6]))
            line = np.genfromtxt(filepath, delimiter=',', filling_values=np.nan, case_sensitive=True,
                                 deletechars='',
                                 replace_space=' ', skip_header=1)
            print(j)
            img.append(np.uint((line[:, index])))
        im = Image.fromarray(np.array(img), "L")
        im.save("out/" + col + "2d" + ".tif", format="tiff", )
        imageio.imwrite(uri="out/1" + col + "2d" + ".tif", im=np.array(img), format="tiff", )
        np.savetxt("out/" + col + "2d" + ".txt", img, fmt='%i', delimiter=',', comments='')


if __name__ == '__main__':

    # get_max_tif()
    # pl3d()
    # apply_filters("5max2d.tif")
    # get_3col_txt_from_txt("5_max2d.txt")
    # img = imageio.imread("1max.tif")

    get_tif_from_csv()

    # for filepath in glob.iglob('out\*2d*.txt'):
    #   get_3col_txt_from_txt(filepath,1.8,1.8,2)
    res = []
    # for i in range (0,1000):
    #     res.append([i,calculate_mkm(i)])
    koef = avg_spectra("led.tif")
    cnt = 0
    for filepath in glob.iglob('tif\*.tif'):
        if filepath == "led.tif": continue
        img = imageio.imread(filepath)
        print(filepath)
        # img = np.delete(img, slice(0, 166, 1), 1)
        # img = np.delete(img, slice(684, -1, 1), 1)

        if (cnt % 10 == 0):
            pass
            #bc(np.transpose(img), cnt)

        cnt += 1
        sum_lines(img, filepath, koef)
    # plt.tight_layout()

    #plt.show()
    #plt.waitforbuttonpress()
    # combine_files()
