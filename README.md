# IB-U-Nets: Learning Robust Prostate Segmentation
We created novel segmentation networks with 3D-IB on top of benchmark U-Net models. To provide an objective comparison between the established CNNs and ours, we created a pipeline that was used for training and testing all models. The repository consists of deep learning models that perform semantic segmentation of the organs from 3D medical images.

## 1. Project Directory ##

### 1.1. Helper Scripts ###
  * [dataloader.py](dataloader.py) - contains data generator. 
  * [models.py](models.py) - fetches required model from the list of networks.
  * [options.py](options.py) - contains default arguments.  
  * [resampler_utils.py](resampler_utils.py) - contains functions for resampling volumes.  
  * [utils.py](utils.py) - loss functions, metrics, data augmentations etc.

### 1.2. Pre-Processing  ###
The [pre_processing](pre_processing) sub-directory consists of the scripts for the following models:
  * [reformatMSDprostate.py](pre_processing/reformatMSDprostate.py) - Change the Task05Prostate MRI data to the required format. The spacing and the direction of the volumes can also be resampled.
  * [reformatPROMISEprostate.py](pre_processing/reformatPROMISEprostate.py) - Change the PROMISE-12 MRI data to the required format. The spacing and the direction of the volumes can also be resampled.
  * [createSubSamples.py](pre_processing/createSubSamples.py) - Create sub-samples from the larger dataset.
  * [add_noise_to_samples.py][add_noise_to_samples.py](pre_processing/createSubSamples.py) - Add artifacts to the required samples.

### 1.3. Models ###
The [networks](networks) sub-directory consists of the scripts for the following models:
1. [IB_3D_Kernels](networks/IB_3D_Kernels.py) - Computes the kernel filters with given shape and parameters
2. U-Net - [3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation.](https://arxiv.org/abs/1606.06650)
3. [IB U-Net](networks/ib_unet.py) - Inductive bias (IB) pathways embedded in the U-Net. Two models exist based on the kernel size ib_unet_k3 (k=3) and ib_unet_k5 (k=5).
4. Attention U-Net - [Attention U-Net: Learning Where to Look for the Pancreas.](https://arxiv.org/abs/1804.03999)
5. [IB Attention U-Net](networks/ib_att_unet.py) - Inductive bias (IB) pathways embedded in the Attention U-Net. Two models exist based on the kernel size ib_att_unet_k3 (k=3) and ib_att_unet_k5 (k=5).
6. SegResNet - [3D MRI brain tumor segmentation using autoencoder regularization](https://arxiv.org/abs/1810.11654)
7. [IB SegResNet](networks/ib_segresnet.py) - Inductive bias (IB) pathways embedded in the SegResNet. Two models exist based on the kernel size ib_segresnet_k3 (k=3) and ib_segresnet_k5 (k=5).

### 1.4. Executable Scripts ###
* [hyperparam_optimization.py](hyperparam_optimization.py) - implementation of Hpbandster to retrieve best training parameters.
* [train.py](train.py) - training program.
* [collect_cv_files.py](collect_cv_files.py) - collates the validation files of each fold after all k-fold training.
* [test.py](test.py) - predicts the segmentation for the test dataset, can also generate noisy data to check robustness.

## 2. Usage ##
* Download/clone the repository.
* Navigate to the project directory. 
```
cd ib_u_nets
```

### 2.1. Dependencies ###
* We trained and tested all our models on Ubuntu 18.04.
* It is recommended to create a virtual environment (preferably Conda) with python3 and above. We use [conda virtual environments](https://www.anaconda.com/distribution/#download-section) to run Python scripts. 
* Make sure to have the latest pip version. Run the following command to retrieve it:
```
pip install --upgrade pip
```

* The codes were written in **python3.8** with PyTorch as the main package, along with other additional packages listed in the [Requirements.txt](Requirements.txt) file. These packages can be installed by running the command: 
```
python -m pip install -r Requirements.txt
``` 
OR 
```
pip install -r Requirements.txt
```

### 2.2. Data Preparation, Generation and Augmentation ###
* The data should be stored in a folder where each input file and corresponding ground truth label is saved in a sub-folder (one folder for one patient).
* The input CT/MRI and ground truth files should be named `data.nrrd` and `label.nrrd` respectively. The volumes should be in the [nrrd](http://teem.sourceforge.net/nrrd/format.html) format.

* We give you two scripts for two difference datasets data to make sure they have the correct format: 
    1. MSD-prostate, and
    2. PROMISE-12.

* Reformat from scratch:
    * Download the MSD-prostate (from here: http://medicaldecathlon.com) or PROMISE-12 dataset (from here: https://promise12.grand-challenge.org) and extract it. Make sure to remove all the hidden data such as `Task05_Prostate.imagesTr/._prostate_04.nii.gz`.
    * Use [reformatMSDprostate.py](pre_processing/reformatMSDprostate.py) or [reformatPROMISEprostate.py](pre_processing/reformatPROMISEprostate.py) script to convert the data samples into the recommended format. 
    * A JSON file is automatically created when the reformatting scripts are executed. It contains complete information about the volumes.
    * Conversion Instance: 
    ```
    python pre_processing/reformatMSDprostate.py --src_folder Task05_Prostate --dst_folder data/train_and_test --change_spacing --new_spacing 0.625 0.625 3.6 --print_info
    ```  

* If the end goal is cross-validation, make sure all the samples are in the same folder structure as shown below. 
```
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
``` 
* However, if you want to train with the full dataset and test on external datasets, then it is mandatory to have separate sub-folders. In this case, the training files have to be resampled to the target spacings. However, the test samples can (recommended) be in their original spacings, and the format could be either `.nii.gz` or `.nrrd`. An example of the required structure of the data folder is shown below: 
```
    dst_folder/
    ├── train/
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
    |
    ├── test_set_1/
    │   ├── 000_data.nrrd
    │   ├── 001_data.nrrd
    │   ├── 002_data.nrrd
    |      ...
    |
    ├── test_set_2/
    │   ├── prostate_00.nii.gz
    │   ├── prostate_01.nii.gz
    │   ├── prostate_02.nii.gz
    |      ...
``` 
* When the sample size is small, like in the case of [Medical Segmentation Decathlon - Prostate Segmentation](http://medicaldecathlon.com/#tasks) dataset, it is better to perform cross-validation such as k-fold. The framework performs 5-fold and 8-fold cross validations as required.

* In such scenarios, the [dataloader.py](dataloader.py) will handle the train-test split based on the `k_fold` and `fold` options. 

* Generating subsamples:
  * Creating a subset from the the large dataset is simple using the [createSubSamples.py](pre_processing/createSubSamples.py) script.
  * Subsample creation Instance: 
    ```
    python pre_processing/createSubSamples.py --source MSD_reformatted --dest MSD/subsamples8 --count 8 --seed 60
    ```  
  
* The [dataloader.py](dataloader.py) reads the MR images, performs augmentation using [utils.py](utils.py) and supplies `n_batches` of data during training.

### 2.3. Arguments (Options) ###
* The pipeline provides several arguments to choose from, that can be found in [options.py](options.py) file. Nonetheless, we have listed a few important options to consider if you are new to this.
* Important arguments/options that have for both hyperparameter search and training:
```
[--img_mod IMG_MOD]                        The modality of the data samples: Choices = [CT, MR]. [Default: MR]
[--patch_shape PATCH_SHAPE]                The voxel shape (height, width, depth) of the data patch. [Default: (176,176,16)]
[--input_channels INPUT_CHANNELS]          The number of channels in the input data. [Default: 1]
[--output_channels OUTPUT_CHANNELS]        The number of channels in the ground truth labels. [Default: 1]
[--n_batches N_BATCHES]                    The number of samples per mini-batch. [Default: 2]
[--init_filters INIT_FILTERS]              The number of kernels/filters in the first encoder block that doubles every block forward. [Default: 32]
[--p_foreground P_FOREGROUND]              The percentage of random CT/MRI patches (corresponding ground truth is empty) sampled per batch. [Default: 0.85]
[--results_path RESULTS_PATH]              The location to save the results of the current experiment. [Default: results/train/]  
[--data_root DATA_ROOT]                    The location of the training (and validation) samples. [Default: data/train/]
[--model_name MODEL_NAME]                  The model architecture for the experiment. Choices=[unet, ib_unet_k3, ib_unet_k5, attention_unet, ib_att_unet_k3, ib_att_unet_k5, segresnet, ib_segresnet_k3, ib_segresnet_k5]. [Default: unet]
[--device_id DEVICE_ID]                    The ID of the GPU to be used. If no GPU, then CPU will be used. [Default: 0]
[--k_fold K_FOLD]                          Select either simple 5-fold or 8-fold cross validation.  Choices = [5,8]. [Default: 8].
[--fold FOLD]                              Select the fold index for the chosen cross validation. Choices = [0,1,2,3,4,5,6,7]. [Default: 0].
[--no_clip NO_CLIP]                        If this is not used, then the MRI data will be clipped to 5 and 95 percentiles.
```
* Models list
 ```
 unet, ib_unet_k3, ib_unet_k5, attention_unet, ib_att_unet_k3, ib_att_unet_k5, segresnet, ib_segresnet_k3, ib_segresnet_k5
 ``` 

* Patch shape
```
MSD-prostate = 288x288x16
PROMISE-12 = 192x192x16
```

### 2.4. Hyperparameter Search ###
* It is not necessary to perform a hyperparameter search. However, if needed, optimization from scratch can be done using [hyperparam_optimization.py](hyperparam_optimization.py) script. 
* Currently, the script searches the best configuration for `beta1` (range: [0.1-0.9]), `dropout_rate` (range: [0.0-0.7]), `learning_rate` (range: [0.000001-0.1])`n_kernels` (range: [16-48]) and `weight_decay` (range: [0.000001-0.0001]). These can be changed as required by modifying their respective minimum and maximum option values. More information can be found [here](https://automl.github.io/HpBandSter/build/html/auto_examples/example_5_pytorch_worker.html).
* Additional arguments for hyperparameter search that could be considered: 
```
[--max_budget MAX_BUDGET]       The maximum number of epochs to train the model for a given configuration. [Default: 500]  
[--min_budget MIN_BUDGET]       The minimum number of epochs to train the model for a given configuration. [Default: 100]  
```

#### 2.4.1. Search Instance ####
```
python hyperparam_optimization.py --data_root data/train_and_test --model_name unet --results_path results/hp_search/unet --max_budget 300 --min_budget 200 --fold 0 --k_fold 5
```

#### 2.4.2. Outputs ####
`results_path` location will contain the following files for every experiment.
* `training_options.txt` stores the options used for the search.
* `configs.json` contains a dictionary of the different hyper-parameters values used in different experiments (runs).
* `results.json` contains a dictionary of the best results of different experiments.
* `overall_best_net.sausage` is saved when the model achieves the highest mean validation DSC of all the configurations. It contains the state_dict and a few other values. Please refer to [hyperparam_optimization.py](hyperparam_optimization.py) for the complete list.
* `config_N_history.csv` lists the corresponding epoch, mean validation loss and mean validation metrics for every epoch. 'N' corresponds to the run count and the summary of this file can be found in 'N'th row of the `results.json` file.

### 2.5. Model Training ###
* Training from scratch can be done using [train.py](train.py) script. 
* If you completed a hyperparameter search, you may change the following arguments for training:
```
--lr LR]                                 Learning rate for the optimizer. [Default: 0.0001] 
[--beta1 BETA1]                          Beta1 for Adam solver. [Default: 0.9] 
[--loss_fn LOSS_FN]                      The loss function to calculate the error. [Default: binary_cross_entropy]
[--dropout_rate DROPOUT_RATE]            The probability of dropping nodes in the layers. [Default: 0.25]
[--training_epochs TRAINING_EPOCHS]      The number of epochs to train the model. [Default: 500]  
```
#### 2.5.1. Training Instance  ####
```
python train.py --data_root data/train_and_test --results_path results/training/unet/fold_0/ --device_id 0 --model_name unet --k_fold 5 --fold 0 --beta1 0.49975392809845687 --dropout_rate 0.06662696925009508 --lr 0.00961050139856093 
```

#### 2.5.2. Outputs ####
`results_path` location will contain the following files for every experiment.
* `training_options.txt` stores the options used for the experiment.
* `history.csv` lists the corresponding epoch, mean training loss, mean training metrics, mean validation loss and mean validation metrics for every epoch.
* `best_net.sausage` is saved when the model achieves the highest mean validation DSC for an epoch. It contains the state_dict and a few other values. Please refer to [train.py](train.py) for the complete list.
* `latest_net.sausage` contains the most recent model that is saved after every 25 epochs.
* `└──validation_predictions` folder save predictions of the validation samples in the modified format (uniform voxel spacing).
* `└──validation_predictions_postprocessed` folder save predictions of the validation samples after processing in the original format (original voxel spacing).
* `val_results_detailed.csv` has detailed DSC value of each validation sample.
* `val_results.csv` has mean, median, std, min and max DSC values of all the validation samples.

#### 2.5.3. Further training ####
* Additional arguments for `train.py` that could considered to resume train or transfer learning: 
```
[--resume_train RESUME_TRAIN]            Continue training from the last saved point. [Default: False] 
[--training_epochs TRAINING_EPOCHS]      The number of epochs to train the model. [Default: 500]  
[--results_path RESULTS_PATH]            The location to first retrieve the results of the previous experiment, and then also store the output of the current run . [Default: results/train/]  
```
`Note:` Resuming training or transfer learning takes results (model files, csv files) from the folder of the previous run and modifies them. It is recommended to keep a separate copy of the original run. Also `total epochs for further training = training_epochs - starting_epochs`

#### 2.5.4. Cross-validation ####
* Complete all the required folds (5 or 8), and ensure that the folder `validation_predictions_postprocessed` and file `val_results_detailed.csv` are present in each fold.
* Collect all the validation samples using the `collect_cv_files.py` script.
```
python collect_cv_files.py --data_root data/train_and_test --exp_path results/training/unet/ --k_fold 5  
```
* This will create a folder `cv_files_postprocessed` inside the `exp_path` folder with all the samples.

`NOTE:` `exp_path` contains all the folds and CV files. `results_path` points to the outputs of a single fold.

### 2.6. Model Testing ###
* Predicting the segmentation results for the test data [test.py](test.py) script. 
* By providing the location of the test dataset, the previously trained model CV folds, segmentations are predicted and saved.
```
[--exp_path EXP_PATH]                      The location of cross-validation results. [Default: results/training/unet]  
[--test_results_path TEST_RESULTS_PATH]    The folder to save the results of the predictions. [Default: results/test/unet] 
[--data_root DATA_ROOT]                    The location of the test samples. [Default: data/test/]
[--k_fold K_FOLD]                          Select either simple 5-fold or 8-fold cross validation.  Choices = [5,8]. [Default: 8].
[--device_id DEVICE_ID]                    The ID of the GPU to be used. If no GPU, then CPU will be used. [Default: 0]
```

#### 2.6.1. Testing Instance  ####
```
python test.py --data_root data/test --exp_path results/testing/unet/ --k_fold 5
```

#### 2.6.2. Outputs ####
`test_results_path` location will contain the following items for every experiment.
* `└──test_predictions` folder will contain predicted labels of all the test samples in the framework format (modified version).
* `└──test_predictions_postprocessed` folder will contain predicted labels of all the test samples after final processing is done. These volumes will have their original format (size, spacings, direction, origin etc.) and original names.

#### 2.6.3. Robustness Evaluation  ####
* Use the file [add_noise_to_samples.py](add_noise_to_samples.py) to add artifacts to the 3D volumes. The three types of artifacts are:
    * Random Gaussian Noise - std values we used for our experiments = 45
    * Gaussian Blur Noise - std values we used for our experiments = 2
    * Motion Artifacts - # transforms we used for our experiments = 1
* We added artifacts to all the volumes in the dataset since we used cross-validation. Use the following arguments to generate a noisy dataset.
```
[--noise NOISE]                               Select the type of noise to add. Choices = [none, random, motion, blur]. [Default: none]
[--blur_std BLUR_STD]                         The amount of standard deviation to be applied to create blur noise. [Default: 2]
[--random_std RANDOM_STD]                     The amount of standard deviation to be applied to create random Gaussian noise. [Default: 45]
[--motion_transforms MOTION_TRANSFORMS]       The number of transforms to be applied to create motion noise. [Default: 1]
```
* An instance is provided below:
```
python add_noise_to_samples.py --data_root data/test --noise blur --blur_std 2
```

## 3. IB-nnU-Net ##
* The following section describes the necessary steps to extended the nnU-Net framework with the inductive bias (IB) kernels. 
* Download/clone the `nnU-Net` repository from its [GitHub page](https://github.com/MIC-DKFZ/nnUNet).
* Navigate to the project directory. 
```
cd nnUNet
```
* Do not install the framework library yet: do not run this `pip install -e .` command yet.

### 3.1. Extra files for implementing IB-nnU-Net ###
* Two files in the folder `IB_nnU_Net` are needed for this:
  1. ib_UNet.py
  2. nnUNetTrainerV2_IB.py
* The files have to be added to the working repository in the correct folders for this to work:
  1. Move the `ib_UNet.py` file into this folder: `nnunet/network_architecture/`.
  2. Move the `nnUNetTrainerV2_IB.py` file into this folder: `nnunet/training/network_training/`.
* Now install the framework library using the `pip install -e .` command.
* That is it! Now you are ready to use the `IB-nnU-Net` model using the `nnUNetTrainerV2_IB` trainer.