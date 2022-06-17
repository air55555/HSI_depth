import numpy as np
import click

@click.command()
@click.pass_context
@click.option('--dir', help='Directory with tiffs, MIN- prefix for finding newest ', required=False, metavar='PATH')
@click.option('--num', help='Number of lines in each dir ', required=False, type=int)
@click.option('--get_only_tiff',
              help='set to 0 to make csv and everyting , 1- process existing csvs, 555- get final tiff from existing csvs, one time',
              required=False, type=int)
@click.option('--command', help='Command of files after which final 3d pic should be displayed  ', required=False,
              type=str)
@click.option('--out_file', help='External img(tif,jpg) that  should be displayed in 3d. Use prefix ARRAY- to load csv files   ', required=False, type=str,
              default="")
@click.option('--show', help='Show 3d pics . 0- no show , 1- show 3d, no mesh, 2 -show and make mesh ,3-show nothing , make meshs', required=False, type=int)

@click.option('--show_cyl', help='Show 3d cylind . 0 - show all , 1 -show cylinder only, 2 - show flat only', required=False, type=int)
@click.option('--file', help='Show 3d cylind . 0 - show all , 1 -show cylinder only, 2 - show flat only', required=False, type=str)


def func(
        ctx: click.Context,
        dir: str,
        num: int,
        command: str,
        get_only_tiff: int,
        show: int,
        show_cyl: int,
        out_file : str,
        file:str


):
    if command == "combine":
        img =  np.genfromtxt(dir + '/' + file, delimiter=',', filling_values=np.nan, case_sensitive=True,
                         deletechars='',
                         replace_space=' ', skip_header=0)
        img = np.concatenate((img,[[0]*img.shape[1]]),axis=0)
        a_out=np.array([])
        for line in range(1,num+1):

            slice= int((img.shape[0]+1)/num)
            part = img[(line-1)*slice:(line)*slice]

            part = np.transpose(part)

            #for i in range(0,part.size):
           #    part(i)

            if line == 1:
                a_out = part
            else:
                part = part[244:]
                a_out=np.concatenate((a_out,part),axis=0)
        f = dir + "/"+ out_file
        np.savetxt(f, a_out, fmt='%i', delimiter=',', comments='')


if __name__ == '__main__':
    import ctypes

    # print("Before: {}".format(ctypes.windll.msvcrt._getmaxstdio()))
    ctypes.windll.msvcrt._setmaxstdio(2048)
    # print("After: {}".format(ctypes.windll.msvcrt._getmaxstdio()))

    # show3d(
    #    'C:/Users\LRS\PycharmProjects\HSI_depth/2021-10-06-15-38-43.5490766_500/00001_X0_704-Y0_584out\mkm_scipy702d.tif') mkm_scipy70_1,4-5-1_3col.csv
    func()
