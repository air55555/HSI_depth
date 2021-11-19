import os
import shutil
import glob

import time





def copy_files(abs_dirname,N, target):
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
            subdir_name = target
            #os.path.join(abs_dirname, str(int(i / N + 1)).zfill(5))
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
        print(f)
        time.sleep(delay)
    #np.savetxt(abs_dirname+"/filelist.txt", file_list, fmt='%s', delimiter=',', comments='')

    return out

if __name__ == '__main__':
    source = "s:/templ"
    target = "s:/camera_out"
    delay = 0
    copy_files(source,500,target)