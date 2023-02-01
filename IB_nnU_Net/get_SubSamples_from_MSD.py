#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.configuration import default_num_threads
from nnunet.utilities.file_endings import remove_trailing_slash
import os, json, shutil
from collections import OrderedDict
import numpy as np
import SimpleITK as sitk

def split_4d_nifti(filename, output_folder, label=False, channels=None, combine_labels=False):
    img_itk = sitk.ReadImage(filename)
    dim = img_itk.GetDimension()
    file_base = filename.split("/")[-1]
    if dim == 3:
        if label:
            if combine_labels:
                spacing = img_itk.GetSpacing()
                origin = img_itk.GetOrigin()
                direction = img_itk.GetDirection()
                img_npy = sitk.GetArrayFromImage(img_itk)
                combined_img_np = np.where(img_npy>0,1,0)
                img_itk_new = sitk.GetImageFromArray(combined_img_np)
                img_itk_new.SetSpacing(spacing)
                img_itk_new.SetOrigin(origin)
                img_itk_new.SetDirection(direction)
                sitk.WriteImage(img_itk_new, join(output_folder, file_base))
            else:
                shutil.copy(filename, join(output_folder, file_base))
        else:
            shutil.copy(filename, join(output_folder, file_base[:-7] + "_0000.nii.gz"))
        return
    elif dim != 4:
        raise RuntimeError("Unexpected dimensionality: %d of file %s, cannot split" % (dim, filename))
    else:
        img_npy = sitk.GetArrayFromImage(img_itk)
        spacing = img_itk.GetSpacing()
        origin = img_itk.GetOrigin()
        direction = np.array(img_itk.GetDirection()).reshape(4,4)
        # now modify these to remove the fourth dimension
        spacing = tuple(list(spacing[:-1]))
        origin = tuple(list(origin[:-1]))
        direction = tuple(direction[:-1, :-1].reshape(-1))
        #for i, t in enumerate(range(img_npy.shape[0])):
        for i, t in enumerate(channels):
            img = img_npy[t]
            img_itk_new = sitk.GetImageFromArray(img)
            img_itk_new.SetSpacing(spacing)
            img_itk_new.SetOrigin(origin)
            img_itk_new.SetDirection(direction)
            sitk.WriteImage(img_itk_new, join(output_folder, file_base[:-7] + "_%04.0d.nii.gz" % i))


def split_4d(input_folder, output_folder=None, count=8, combine_labels=False, combined_label_name="foreground"):
    assert isdir(join(input_folder, "imagesTr")) and isdir(join(input_folder, "labelsTr")) and \
           isfile(join(input_folder, "dataset.json")), \
        "The input folder must be a valid Task folder from the Medical Segmentation Decathlon with at least the " \
        "imagesTr and labelsTr subfolders and the dataset.json file"

    # Get the location of the dataset.
    while input_folder.endswith("/"):
        input_folder = input_folder[:-1]

    # Delete old folder if it exits.
    if isdir(output_folder):
        shutil.rmtree(output_folder)

    # Read the original dataset.json file to get all the information.
    jsonFile = open(join(input_folder, "dataset.json"))
    orig_dataset_info = json.load(jsonFile)

    # Create new json dict to save info about the subsamples.
    json_dict = OrderedDict()
    json_dict['name'] = orig_dataset_info['name']
    json_dict['description'] = orig_dataset_info['description']
    json_dict['tensorImageSize'] = orig_dataset_info['tensorImageSize']
    json_dict['reference'] = orig_dataset_info['reference']
    json_dict['licence'] = orig_dataset_info['licence']
    # Spelling mistake in some of the original json files.
    try:
        json_dict['release'] = orig_dataset_info['release']
    except:
        json_dict['release'] = orig_dataset_info['relase']
    
    # Only T2 modalility.
    json_dict['modality'] = {'0': orig_dataset_info['modality']['0']}
    if combine_labels:
        json_dict['labels'] = {'0': orig_dataset_info['labels']['0'],
                               '1': str(combined_label_name)} 
    else:
        json_dict['labels'] = orig_dataset_info['labels']
    
    # Now test files required.
    json_dict['test'] = []
    json_dict['numTest'] = 0

    # For now.
    json_dict['training'] = []
    json_dict['numTraining'] = count

    # Retrieving subsample indices.
    indices = np.arange(orig_dataset_info['numTraining'])
    np.random.seed(2022)
    shuffled_indices = np.random.permutation(indices)
    subSample_indices = shuffled_indices[:count]
    
    # Dict of original file names.
    orig_train_files = orig_dataset_info['training']
    
    maybe_mkdir_p(output_folder)
    maybe_mkdir_p(os.path.join(output_folder, "imagesTr"))
    maybe_mkdir_p(os.path.join(output_folder, "labelsTr"))

    channel_list = []
    for channel_key in json_dict['modality'].keys():
        channel_list.append(int(channel_key))

    for idx, train_file in enumerate(orig_train_files):
        if idx in subSample_indices:
            json_dict['training'].append(train_file)
            
            split_4d_nifti(input_folder + train_file['image'][1:], 
                           os.path.join(output_folder, "imagesTr"), 
                           False,
                           channel_list,
                           False)
            split_4d_nifti(input_folder + train_file['label'][1:], 
                           os.path.join(output_folder, "labelsTr"), 
                           True,
                           None,
                           combine_labels)
    
    save_json(json_dict, os.path.join(output_folder, "dataset.json"))

def crawl_and_remove_hidden_from_decathlon(folder):
    folder = remove_trailing_slash(folder)
    assert folder.split('/')[-1].startswith("Task"), "This does not seem to be a decathlon folder. Please give me a " \
                                                     "folder that starts with TaskXX and has the subfolders imagesTr, " \
                                                     "labelsTr and imagesTs"
    subf = subfolders(folder, join=False)
    assert 'imagesTr' in subf, "This does not seem to be a decathlon folder. Please give me a " \
                                                     "folder that starts with TaskXX and has the subfolders imagesTr, " \
                                                     "labelsTr and imagesTs"
    assert 'imagesTs' in subf, "This does not seem to be a decathlon folder. Please give me a " \
                                                     "folder that starts with TaskXX and has the subfolders imagesTr, " \
                                                     "labelsTr and imagesTs"
    assert 'labelsTr' in subf, "This does not seem to be a decathlon folder. Please give me a " \
                                                     "folder that starts with TaskXX and has the subfolders imagesTr, " \
                                                     "labelsTr and imagesTs"
    _ = [os.remove(i) for i in subfiles(folder, prefix=".")]
    _ = [os.remove(i) for i in subfiles(join(folder, 'imagesTr'), prefix=".")]
    _ = [os.remove(i) for i in subfiles(join(folder, 'labelsTr'), prefix=".")]
    _ = [os.remove(i) for i in subfiles(join(folder, 'imagesTs'), prefix=".")]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="The MSD provides data as 4D Niftis with the modality being the first"
                                                 " dimension. We think this may be cumbersome for some users and "
                                                 "therefore expect 3D niftixs instead, with one file per modality. "
                                                 "This utility will convert 4D MSD data into the format nnU-Net "
                                                 "expects")
    parser.add_argument("-i", help="Input folder. Must point to a TaskXX_TASKNAME folder as downloaded from the MSD "
                                   "website", required=True)
    parser.add_argument("-p", required=False, default=default_num_threads, type=int,
                        help="Use this to specify how many processes are used to run the script. "
                             "Default is %d" % default_num_threads)
    parser.add_argument("-o", required=True, default=None, type=str,
                        help="The output folder.")
    parser.add_argument("-c", required=False, default=8, type=int,
                        help="Number of files in the subsamples.")
    parser.add_argument("-combine_labels", required=False, default=False, action="store_true",
                        help="Combine all the ground-truth values greater than 0 into 1.")
    parser.add_argument("-combined_label_name", required=False, default="foreground", type=str,
                        help="The name of the label after combining all foreground values to a single value.")
    args = parser.parse_args()

    crawl_and_remove_hidden_from_decathlon(args.i)

    split_4d(args.i, args.o, args.c, args.combine_labels ,args.combined_label_name)


if __name__ == "__main__":
    main()
