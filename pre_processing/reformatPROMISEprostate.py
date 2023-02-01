# DOCUMENT INFORMATION
'''
    Project Name: IB U-Nets
    File Name   : reformatPROMISEprostate.py
    Code Author : Shrajan Bhandary
    Created on  : 23 July 2021
    Program Description:
        This script converts the Prostate-12 data and its ground-truth labels to the required format.
        The final data should will have the following folder structure. Uses B-spline interpolation 
        for the MR images and nearest neighbour interpolation for the ground-truth labels.

        The final data should will have the following folder structure.

            dst_folder/
            ├── train_and_test/
            |   └── original_gts/
            │      ├── 000_label.nrrd
            │      ├── 001_label.nrrd
            |      ...
            │   └── 000/
            │      ├── data.nrrd
            │      ├── label.nrrd
            |      ...
            │   └── 001/
            │      ├── data.nrrd
            │      ├── label.nrrd
            |       ...
            |   ├──dataset_info.json
            
    Versions:
    |----------------------------------------------------------------------------------------|
    |-----Last modified-----|----------Author----------|---------------Remarks---------------|
    |----------------------------------------------------------------------------------------|
    |    23 July 2021       |     Shrajan Bhandary     |  Implemented necessary functions.   |
    |    21 January 2022    |     Shrajan Bhandary     | Cleaned up stuff and added comments.|
    |----------------------------------------------------------------------------------------|
'''

# LIBRARY IMPORTS

import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
import argparse, os
from resampler_utils import resample_3D_image
from collections import OrderedDict
import numpy as np

# IMPLEMENTATION

if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_folder", type=str, default="PROMISE_12", required=False, help="Folder/directory to read the original PROMISE-12 prostate data.")
    parser.add_argument("--dst_folder", type=str, default="mri_framework/train_and_test", required=False, help="Folder/directory to save the converted data.")
    parser.add_argument("--change_spacing", action='store_true', help="If set, then data and corresponding label will be resampled to new_spacing.")
    parser.add_argument("--new_spacing", type=float, nargs=3, default=(0.6125, 0.6125, 3.6), required=False, help="Spacing to be resampled.")
    parser.add_argument("--change_direction", action='store_true', help="If set, then direction of data and corresponding label changed.")
    parser.add_argument("--print_info", action='store_true', help="If set, then some information about the volumes will be printed for every patient.")
    opt = parser.parse_args()
    
    # File Paths
    src_folder = opt.src_folder
    dst_folder = opt.dst_folder
    origial_gts_path = os.path.join(dst_folder, "original_gts")

    # Create output folders.
    maybe_mkdir_p(dst_folder)
    maybe_mkdir_p(origial_gts_path)
    
    # Initialize an ordered dictionary to store contents in to the JSON file.
    json_dict = OrderedDict()
    json_dict['name'] = "PROMISE-12"                        # Name of the dataset.
    json_dict['description'] = "prostate MR images"         # Description.
    json_dict['modality'] = "MR"                            # Modality of the volumes.
    json_dict['files'] = dict()                             # To store each patient information.

    # Get the paths of the data and labels.
    segmentations = subfiles(src_folder, suffix="segmentation.mhd")
    raw_data = [i for i in subfiles(src_folder, suffix="mhd") if not i.endswith("segmentation.mhd")]

    # Patient identification number.
    p_id = 0

    # Resample, change format to nrrd and save data and label in individually folders.
    for index, (dataPath, labelPath) in enumerate(zip(raw_data, segmentations)):
        p_id_str = str(p_id).zfill(3)
        maybe_mkdir_p(join(dst_folder,str(p_id_str)))
        
        old_data = sitk.ReadImage(dataPath)
        old_label = sitk.ReadImage(labelPath)

        # Resample if necessary.
        if old_data.GetSpacing() != opt.new_spacing or old_data.GetDirection() != (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
            new_data = resample_3D_image(sitkImage=old_data, newSpacing=opt.new_spacing,
                        interpolation="BSpline", change_spacing=opt.change_spacing, change_direction=opt.change_direction)
                
            new_label = resample_3D_image(sitkImage=old_label, newSpacing=opt.new_spacing, 
                        interpolation="nearest", change_spacing=opt.change_spacing, change_direction=opt.change_direction)

        else:
            new_data = old_data
            new_label = old_label

        # Save the original label file in "dst_folder/original_gts" sub-folder.
        orig_label_path = os.path.join("original_gts", os.path.split(labelPath)[-1])
        sitk.WriteImage(sitk.Cast(old_label, sitk.sitkUInt8), 
                        os.path.join(dst_folder,orig_label_path), True) 


        # Save the new data file.
        data_fname = os.path.join(dst_folder,str(p_id_str), "data.nrrd")
        sitk.WriteImage(sitk.Cast(new_data, sitk.sitkFloat32), data_fname, True) 
        
        # Save the new label file.
        label_fname = os.path.join(dst_folder,str(p_id_str), "label.nrrd")
        sitk.WriteImage(sitk.Cast(new_label, sitk.sitkUInt8), label_fname, True)

        # Information to store in the JSON file.
        json_dict['files'][p_id_str] = {
            'patient_id': p_id_str,
            'orig_volume_info': {'dataFilePath': os.path.split(dataPath)[-1],
                                 'labelFilePath': orig_label_path,
                                 'spacing':old_label.GetSpacing(),
                                 'size': old_label.GetSize(),   
                                 'direction': old_label.GetDirection(),
                                 'origin': old_label.GetOrigin(),
                                },
            'new_volume_info':{'dataFilePath': os.path.join(str(p_id_str), "data.nrrd"),
                                'labelFilePath': os.path.join(str(p_id_str), "label.nrrd"),
                                'spacing':new_label.GetSpacing(),
                                'size': new_label.GetSize(),   
                                'direction': new_label.GetDirection(),
                                'origin': new_label.GetOrigin(),
                                },
        }
        
        # Optional print information.
        if opt.print_info:
            print("\nFolder number: " + str(p_id_str) + " " + dataPath + " " + labelPath)
            print("The old data image had shape: " + str(old_data.GetSize()) + " with spacing: " + str(old_data.GetSpacing()))
            print("The new data image had shape: " + str(new_data.GetSize()) + " with spacing: " + str(new_data.GetSpacing()))
            print("The old label image has shape: " + str(old_label.GetSize()) + " with spacing: " + str(old_label.GetSpacing()))
            print("The new label image has shape: " + str(new_label.GetSize()) + " with spacing: " + str(new_label.GetSpacing()))

        # Increment value of p_id.
        p_id += 1

    # Ensure file size is correct.
    if p_id != len(json_dict['files']):
        print("The file size doesn't match")
        
    # To store final patient count. 
    json_dict['numFiles'] = p_id                               

    # Save the final JSON file with the complete data description.
    save_json(json_dict, os.path.join(dst_folder, "dataset_info.json"))