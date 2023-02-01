"""
Creates subsamples of size "count" (count < n) from the full dataset. To reproduce results or 
ensure all volumes of smaller subsample are retained in the larger subsample, use the same 
"seed" value. Before creating subsamples make sure that volumes are in the rqeuired format, 
i.e., MRIs: data.nrrd and Ground-truths: label.nrrd.
"""
import argparse, os
from shutil import copy2
import numpy as np

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=8, required=False, help='Number of files to copy')
    parser.add_argument('--source', type=str, default='fulldataset', required=False, help='Location of the files.')
    parser.add_argument('--dest', type=str, default='subsamples', required=False, help='Location to save the results.')
    parser.add_argument('--seed', type=int, default=60, required=False, help='If set, then uses seeding')

    opt = parser.parse_args()

    folders = os.listdir(opt.source)
    
    if opt.seed > 0:
        print("Seeding")
        np.random.seed(opt.seed)
        data_idx = np.random.permutation(folders)
    else:
        print("No Seeding")
        data_idx = np.random.permutation(folders)
    
    totalFiles = 0

    for index in range(0,opt.count):

        out_folder = str(data_idx[index])
        
        if not os.path.exists(os.path.join(opt.dest, out_folder)):
            os.makedirs(os.path.join(opt.dest, out_folder))

        dataSource = os.path.join(opt.source, out_folder, "data.nrrd")
        dataDest = os.path.join(opt.dest, out_folder, "data.nrrd")
        copy2(dataSource, dataDest)

        labelSource = os.path.join(opt.source, out_folder, "label.nrrd")
        labelDest = os.path.join(opt.dest, out_folder, "label.nrrd")
        copy2(labelSource, labelDest)

        totalFiles +=1

    print(f"Total files in the folder = {totalFiles}")

