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
from collections import OrderedDict
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
import argparse
import numpy as np

def export_for_submission(source_dir, target_dir):
    """
    promise wants mhd :-/
    :param source_dir:
    :param target_dir:
    :return:
    """
    files = subfiles(source_dir, suffix=".nii.gz", join=False)
    target_files = [join(target_dir, i[:-7] + ".mhd") for i in files]
    maybe_mkdir_p(target_dir)
    for f, t in zip(files, target_files):
        img = sitk.ReadImage(join(source_dir, f))
        sitk.WriteImage(img, t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_folder", type=str, default="preds", required=False, help="Folder to get files.")
    parser.add_argument("--dst_folder", type=str, default="labels" , required=False, help="Folder to save files.")
    parser.add_argument("--count", type=int, default=8, required=False, help="Number of files in the subsamples.")
    opt = parser.parse_args()
    folder = opt.src_folder
    out_folder = opt.dst_folder

    maybe_mkdir_p(join(out_folder, "imagesTr"))
    maybe_mkdir_p(join(out_folder, "imagesTs"))
    maybe_mkdir_p(join(out_folder, "labelsTr"))
    # train
    current_dir = folder
    segmentations = subfiles(current_dir, suffix="segmentation.mhd")
    raw_data = [i for i in subfiles(current_dir, suffix="mhd") if not i.endswith("segmentation.mhd")]
    
    indices = np.arange(len(raw_data))
    np.random.seed(2022)
    shuffled_indices = np.random.permutation(indices)
    subSample_indices = shuffled_indices[:opt.count]
    
    subSample_raw_data = []

    for idx, i in enumerate(raw_data):
        if idx in subSample_indices:
            out_fname = join(out_folder, "imagesTr", i.split("/")[-1][:-4] + "_0000.nii.gz")
            sitk.WriteImage(sitk.ReadImage(i), out_fname)
            subSample_raw_data.append(i)

    for idx, i in enumerate(segmentations):
        if idx in subSample_indices:
            out_fname = join(out_folder, "labelsTr", i.split("/")[-1][:-17] + ".nii.gz")
            sitk.WriteImage(sitk.ReadImage(i), out_fname)

    json_dict = OrderedDict()
    json_dict['name'] = "PROMISE12"
    json_dict['description'] = "prostate"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "see challenge website"
    json_dict['licence'] = "see challenge website"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "MRI",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "prostate"
    }
    json_dict['numTraining'] = len(subSample_raw_data)
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i.split("/")[-1][:-4], "label": "./labelsTr/%s.nii.gz" % i.split("/")[-1][:-4]} for i in
                             subSample_raw_data]
    json_dict['test'] = []
    json_dict['numTest'] = 0
    save_json(json_dict, os.path.join(out_folder, "dataset.json"))

