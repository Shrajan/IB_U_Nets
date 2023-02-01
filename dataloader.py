# DOCUMENT INFORMATION
'''
    Project Name: IB U-Nets
    File Name   : dataloader.py
    Code Author : Dejan Kostyszyn and Shrajan Bhandary
    Created on  : 14 March 2021
    Program Description:
        This program contains contains training data generator.
            
    Versions:
    |----------------------------------------------------------------------------------------|
    |-----Last modified-----|----------Author----------|---------------Remarks---------------|
    |----------------------------------------------------------------------------------------|
    |    14 March 2021      |     Dejan Kostyszyn      |  Implemented necessary functions.   |
    |    01 August 2021     |     Shrajan Bhandary     |    Changed image reading method.    |
    |    05 August 2021     |     Dejan Kostyszyn      |    Implemented cross-validation.    |
    |    25 August 2021     |     Shrajan Bhandary     | Unit tests to ensure data validity. |
    |    22 January 2022    |     Shrajan Bhandary     | Cleaned up stuff and added comments.|
    |----------------------------------------------------------------------------------------|
'''

# LIBRARY IMPORTS
import os, torch, utils, json, random, nrrd, tqdm, time
import numpy as np
from os.path import join
from natsort import natsorted
import SimpleITK as sitk
torch.manual_seed(2021)
from batchgenerators.utilities.file_and_folder_operations import *
from sklearn.model_selection import KFold

# Faster runtime.
torch.backends.cudnn.benchmark = True

# IMPLEMENTATION

class Dataset(torch.utils.data.Dataset):
    """
    Loads data and corresponding label and returns pytorch float tensor. In detail:
    * Retrieves complete dataset information from the JSON file.
    * Reads file paths from the JSON file.
    * Segregates files into train and val.
    * 
     
    param opt               # Command line arguments from the user or default values from the options file.
    param split_type        # Type of data split: options = ["train", "val"]

    The data should have the following folder structure. Train and test can be in separate folders if
    cross-validation is not being done.

    data_root/
    ├── train_and_val/
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
    
    """
    def __init__(self, opt=None, split_type="train"):
        
        # Options.
        self.opt = opt

        # Data path.
        self.data_root = opt.data_root

        # Type of data split.
        self.split_type = split_type

        # Variable to store the complete information about the dataset.
        self.dataset_info = None

        # Get the required information about the volumes.
        self.patient_ids, self.data_paths, self.label_paths, self.mask_paths = self.read_data_paths()

        # Get data indices based on the split type, k-fold and fold.
        self.determine_split_indices(opt=self.opt, split_type=self.split_type)

        # Determine the data indices of the current split.
        self.split_idx = self.get_split_idx()

    def __len__(self):
        return len(self.split_idx)
        
    def get_split_idx(self):
        if self.split_type == "train":
            return self.train_idx
        elif self.split_type == "val":
            return self.val_idx

    def nr_of_patients(self):
        return self.__len__()

    def get_dataset_info(self):
        """
        Return the complete information about the dataset.
        """
        return self.dataset_info

    def read_data_paths(self):
        """
        Reads data paths after retrieving "dataset_info.json" file.
        """
        
        data_root = self.opt.data_root

        # Reads all the information about the dataset from 'dataset_info.json' file.
        jsonFile = open(join(data_root, "dataset_info.json"))
        self.dataset_info = json.load(jsonFile)

        # Read patient ids.
        patient_ids = list(self.dataset_info["files"].keys())
        patient_ids = natsorted(patient_ids)

        if len(patient_ids) != self.dataset_info["numFiles"]:
            raise Exception("The current number of files and actual count don't match. \
                            Please check the dataset.json file.")

        # Read data and label paths.
        data_paths = []
        label_paths = []
        mask_paths = []

        for p_id in patient_ids:
            data_paths.append(join(data_root, 
                                self.dataset_info["files"][p_id]["new_volume_info"]["dataFilePath"]))
            label_paths.append(join(data_root, 
                                self.dataset_info["files"][p_id]["new_volume_info"]["labelFilePath"]))
            
            # Future work for prostate cancer detection.
            m_path = join(data_root, p_id, "mask.nrrd")
            if os.path.exists(m_path):
                mask_paths.append(m_path)
            else:
                mask_paths.append(None)
            
        return patient_ids, data_paths, label_paths, mask_paths

    def shuffle_patch_choice(self):
        """
        It is randomly decided for which patients only background patches
        shall be returned.
        """
        # Randomly choose 20% of val patches to include only background.
        self.no_prostate_patch_idx = random.sample(list(self.val_idx), int(len(self.val_idx)*0.2))

    def different_spacing(self, spacing_1, spacing_2, tolerance=0.0001):
        """
        Checks whether the spacings match with a tolerance.
        """
        if abs(spacing_1[0]-spacing_2[0]) > tolerance:
            return True
        if abs(spacing_1[1]-spacing_2[1]) > tolerance:
            return True
        if abs(spacing_1[2]-spacing_2[2]) > tolerance:
            return True
        return False

    def determine_split_indices(self, opt=None, split_type="train"):
        """
        Splits the patients into train, validation and train based on the k-fold and fold parameters.
        """
        # Split data into training, validation and testing set.
        self.data_idx = np.arange(len(self.patient_ids))
        np.random.seed(self.opt.seed)

        # If shuffle is required.
        if not opt.no_shuffle == True:
            self.data_idx = np.random.permutation(self.data_idx)

        # No cross-validation.
        if opt.k_fold == 1:
            self.train_size = int(0.8*len(self.patient_ids))
            self.val_size = len(self.patient_ids) - self.train_size
            #self.train_size = -9
            self.train_idx = self.data_idx[:self.train_size]
            self.val_idx = self.data_idx[self.train_size:]

        # 5-fold or 8-fold cross-validation.
        else:  
            kf = KFold(n_splits=int(opt.k_fold))
            for idx, (train_index, test_index) in enumerate(kf.split(self.data_idx)):
                if idx == opt.fold:
                    self.train_idx, self.val_idx = self.data_idx[train_index], self.data_idx[test_index]

    def __getitem__(self, idx):
        """
        Read patient id, data and label and return them.
        """
        current_idx = self.split_idx[idx]
        p_id = self.patient_ids[current_idx]

        # Read data and label from memory.
        data_array, data_header = nrrd.read(self.data_paths[current_idx])
        label_array, label_header = nrrd.read(self.label_paths[current_idx])
        label_array = np.where(label_array>0,1,0)
        #data_array = np.load(self.data_paths[current_idx])
        #label_array = np.load(self.label_paths[current_idx])

        # Clip data to 0.5 and 99.5 percentiles.
        if self.opt.clip == True:
            low, high = np.percentile(data_array, [0.5, 99.5])
            data_array = np.clip(data_array, low, high) 

        # Convert numpy to torch format.
        data_array = torch.FloatTensor(data_array)
        label_array = torch.ByteTensor(label_array)

        if self.split_type == "train":

            # Decide randomly whether to choose ROI or not.
            random_value = torch.rand(1)
            if random_value < self.opt.p_foreground:
                data_array, label_array = utils.select_roi_patches(data=data_array, label=label_array, opt=self.opt)
            else:
                data_array, label_array = utils.select_random_patches(data=data_array, label=label_array, opt=self.opt)

            # Data augmentation.
            if self.opt.no_augmentation == False:
                data_array, label_array = utils.data_augmentation_batch(data_array, label_array, self.opt)

        else:
            # Adding a channel axis.
            data_array = data_array.unsqueeze(0)
            label_array = label_array.unsqueeze(0)

        data_array = utils.normalize(data_array, self.opt)

        return p_id, data_array, label_array
