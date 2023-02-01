# DOCUMENT INFORMATION
'''
    Project Name: IB U-Nets
    File Name   : utils.py
    Code Author : Dejan Kostyszyn and Shrajan Bhandary
    Created on  : 14 March 2021
    Program Description:
        This program contains implementation of utility functions.
            
    Versions:
    |----------------------------------------------------------------------------------------|
    |-----Last modified-----|----------Author----------|---------------Remarks---------------|
    |----------------------------------------------------------------------------------------|
    |    14 March 2021      |     Dejan Kostyszyn      |  Implemented necessary functions.   |
    |    18 April 2021      |     Shrajan Bhandary     |  Changed structure to object type.  |
    |    22 January 2022    |     Shrajan Bhandary     | Cleaned up stuff and added comments.|
    |----------------------------------------------------------------------------------------|
'''

import numpy as np
import matplotlib.pyplot as plt
import os, sys, torch, random, cc3d
from medpy.metric.binary import dc
import SimpleITK as sitk
from pre_processing.resampler_utils import resample_3D_image

import torchio as tio
from torch.optim.lr_scheduler import _LRScheduler
from monai.losses import DiceCELoss

####################################################
#----------------- OPTIONAL STUFF -----------------#
####################################################

def str_to_bool(value):
    """
    Turns a string into boolean value.
    """
    t = ['true', 't', '1', 'y', 'yes', 'ja', 'j']
    f = ['false', 'f', '0', 'n', 'no', 'nein']
    if value.lower() in t:
        return True
    elif value.lower() in f:
        return False
    else:
        raise ValueError("{} is not a valid boolean value. Please use one out of {}".format(value, t + f))

def overwrite_request(path):
    if os.path.exists(path):
        valid = False
        while not valid:
            answer = input("{} already exists. Are you sure you want to overwrite everything in this folder? [yes/no]\n".format(path))
            if str_to_bool(answer):
                valid = True
            elif not str_to_bool(answer):
                sys.exit(1)

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("Created folder(s) {}".format(path))
    else:
        print("Folder(s) {} already exist(s).".format(path))

def plot_losses(opt, path, title, xlabel, ylabel, plot_name, *args, axis="auto"):
    """
    Creates nice plots and saves them as PNG files onto permanent memory.
    """
    fig = plt.figure()
    plt.title(title)
    for element in args:
        plt.plot(element[0], label=element[1], alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axis(axis)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(path, plot_name))
    plt.close(fig)


####################################################
#------------------ DATA LOADING ------------------#
####################################################

def normalize(data, opt):
  """
  Normalize data with method that is defined in opt.
  """
  if opt.normalize == "local":
      data = (data - data.mean() + 1e-8) / (data.std() + 1e-8) # z-score
  elif opt.normalize == "global":
      mean = 0.1876465981086576
      std = 1.3712485720416003
      data = data - mean
      data = data / std
  elif opt.normalize == "-11":
      data -= data.min()
      data /= data.max()
      data = (data - 0.5) / 0.5 # -> [-1, 1]
  elif opt.normalize == "max10":
      data[data>10] = 10
  elif opt.normalize == "max5":
      data[data>5] = 5
  elif opt.normalize.lower() == "none":
      #print(opt.normalize.lower())
      return data
  else:  
      sys.exit("Normalize parameter must be one out of {None, local, global, -11, max10}")
  return data

def get_roi_centroid(seg):
  """
  Input param: 3D binary array.
  Computes the centroid by taking the mittle of the
  outer boundaries. Returns center coordinates as tuple
  (x, y, z)
  """
  nonzeros = torch.nonzero(seg)
  maxX, maxY, maxZ = nonzeros[:, 0].max().item(), nonzeros[:, 1].max().item(), nonzeros[:, 2].max().item()
  minX, minY, minZ = nonzeros[:, 0].min().item(), nonzeros[:, 1].min().item(), nonzeros[:, 2].min().item()
  return (maxX - ((maxX - minX) // 2), maxY - ((maxY - minY) // 2), maxZ - ((maxZ - minZ) // 2))

def get_extreme_coordinates(center_point, axis_patch_shape, axis_end_point):
  if max(int(center_point - axis_patch_shape / 2),0) == 0 : 
    return 0, axis_patch_shape 

  elif min(int(center_point + axis_patch_shape / 2),axis_end_point) == axis_end_point:
    return axis_end_point - axis_patch_shape, axis_end_point

  else:
    leftPoint = int(center_point - axis_patch_shape / 2)
    rightPoint = leftPoint + axis_patch_shape
    return leftPoint, rightPoint

def select_random_patches(data=None, label=None, prostate=None, mask=None, opt=None):
  """
  Generate random patches from 3D tensors
  """
  patch_shape = [opt.patch_shape[0], opt.patch_shape[1], opt.patch_shape[2]]
  if not opt.no_augmentation:
    patch_shape[0] += opt.margin
    patch_shape[1] += opt.margin
    patch_shape[2] += opt.margin

  x, y, z = data.shape[-3], data.shape[-2], data.shape[-1]

  data_batch = torch.zeros((opt.input_channels, patch_shape[0], patch_shape[1], patch_shape[2]))
  label_batch = torch.zeros((opt.output_channels, patch_shape[0], patch_shape[1], patch_shape[2]))

  l_x = torch.randint(low=0, high=x-patch_shape[0], size=(1,), dtype=torch.int16)[0]
  l_y = torch.randint(low=0, high=y-patch_shape[1], size=(1,), dtype=torch.int16)[0]
  l_z = torch.randint(low=0, high=z-patch_shape[2], size=(1,), dtype=torch.int16)[0]

  l_batch = label[l_x:l_x + patch_shape[0], l_y:l_y + patch_shape[1], l_z:l_z + patch_shape[2]]
  d_batch = data[l_x:l_x + patch_shape[0], l_y:l_y + patch_shape[1], l_z:l_z + patch_shape[2]]
  
  label_batch[0, ...] = l_batch
  data_batch[0, ...] = d_batch

  if opt.input_channels == 2:
      data_batch[1, ...] = prostate[l_x:l_x + patch_shape[0], l_y:l_y + patch_shape[1], l_z:l_z + patch_shape[2]]

  if opt.input_channels == 3:
      data_batch[1, ...] = prostate[l_x:l_x + patch_shape[0], l_y:l_y + patch_shape[1], l_z:l_z + patch_shape[2]]
      data_batch[2, ...] = mask[l_x:l_x + patch_shape[0], l_y:l_y + patch_shape[1], l_z:l_z + patch_shape[2]]

  return data_batch, label_batch

def select_roi_patches(data=None, label=None, prostate=None, mask=None, opt=None):
  """
  Generate region of interest patches from 3D tensors
  """
  patch_shape = [opt.patch_shape[0], opt.patch_shape[1], opt.patch_shape[2]]
  if not opt.no_augmentation:
    patch_shape[0] += opt.margin
    patch_shape[1] += opt.margin
    patch_shape[2] += opt.margin

  x, y, z = data.shape[-3], data.shape[-2], data.shape[-1]

  data_batch = torch.zeros((opt.input_channels, patch_shape[0], patch_shape[1], patch_shape[2]))
  label_batch = torch.zeros((opt.output_channels, patch_shape[0], patch_shape[1], patch_shape[2]))

  if opt.input_channels > 1:
    c_x, c_y, c_z = get_roi_centroid(prostate)
  else:  
    c_x, c_y, c_z = get_roi_centroid(label)
    
  l_x, r_x = get_extreme_coordinates(c_x, patch_shape[0], x)
  l_y, r_y = get_extreme_coordinates(c_y, patch_shape[1], y)
  l_z, r_z = get_extreme_coordinates(c_z, patch_shape[2], z)

  l_batch = label[l_x:r_x , l_y:r_y, l_z:r_z]
  d_batch = data[l_x:r_x , l_y:r_y, l_z:r_z]

  label_batch[0, ...] = l_batch
  data_batch[0, ...] = d_batch

  if opt.input_channels == 2:
      data_batch[1, ...] = prostate[l_x:r_x , l_y:r_y, l_z:r_z]

  if opt.input_channels == 3:
      data_batch[1, ...] = prostate[l_x:r_x , l_y:r_y, l_z:r_z]
      data_batch[2, ...] = mask[l_x:r_x , l_y:r_y, l_z:r_z]
  
  return data_batch, label_batch

def data_augmentation_batch(data_batch, label_batch, opt):
  """
  With a probability of 50% perform elastic deformation
  and with probability of 50% perform flip over x-axis.
  """
  # Define input subject.
  input_subject = tio.Subject({'data': tio.ScalarImage(tensor=data_batch), 'label': tio.LabelMap(tensor=label_batch)})

  # Define flip transform.
  flip_transforms = {
    tio.RandomFlip(axes=('LR',), flip_probability=0.25), # Lateral flip
    tio.RandomFlip(axes=('AP',), flip_probability=0.05) # Anterior posterior flip
  }
  # Define spatial transformations.
  spatial_transforms = {
    tio.RandomAffine(
      scales=(0.8, 1.2),
      degrees=5,
      translation=(-5,5),
      isotropic=False,
      center='image',
      default_pad_value='mean',
      image_interpolation='linear'
    ): 0.25,
    tio.RandomAffine(
      scales=(0.8, 1.2),
      degrees=5,
      translation=(-5,5),
      isotropic=False,
      center='image',
      default_pad_value='mean',
      image_interpolation='bspline'
    ): 0.25,
    tio.RandomElasticDeformation(
      num_control_points=4,
      max_displacement=2,
      locked_borders=1,
      image_interpolation='linear'
    ): 0.25,
    tio.RandomElasticDeformation(
      num_control_points=4,
      max_displacement=2,
      locked_borders=1,
      image_interpolation='bspline'
    ): 0.25
  }

  noise_transforms = {
    tio.transforms.RandomNoise(std=15): 0.25,
    tio.transforms.RandomMotion(num_transforms=2): 0.25,
    tio.transforms.RandomBlur(std=2.75): 0.25,
    tio.transforms.RandomGamma(log_gamma=(-0.1, 0.1)): 0.25,
  }

  # Compose transforms.
  transforms = tio.Compose([
    tio.OneOf(flip_transforms, p=0.10),
    tio.OneOf(spatial_transforms, p=0.10),
    tio.OneOf(noise_transforms, p=0.10),
  ])

  # Perform transformations.
  output_subject = transforms(input_subject)

  data_batch = output_subject['data'].data
  label_batch = output_subject['label'].data

  return data_batch, label_batch


####################################################
#-------------- TRAINING PARAMETERS ---------------#
####################################################

def set_optimizer(opt, model_params):
    if opt.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model_params, lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps, weight_decay=opt.weight_decay)
    elif opt.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(model_params, lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps, weight_decay=opt.weight_decay)
    elif opt.optimizer.lower() == "adamax":
        optimizer = torch.optim.Adamax(model_params, lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps, weight_decay=opt.weight_decay)
    elif opt.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(model_params, lr=opt.lr, momentum=0.99, weight_decay=opt.weight_decay, nesterov=True)    
    else:
        sys.exit("{} is not a valid optimizer. Choose one of: adam, adamax or sgd".format(opt.optimizer))
    return optimizer

class BinaryDiceLoss(torch.nn.Module):
  """
  Generalized Dice Loss for binary case (Only 0 and 1 in ground truth labels.)
  """
  def __init__(self, smooth=0.000001):
    super(BinaryDiceLoss, self).__init__()
    self.smooth = smooth

  def forward(self, y_pred, y_true):
    y_pred = torch.sigmoid(y_pred)   # The output from the model is logits.

    # Original Dice Loss.
    numerator = 2 * torch.mul(y_pred, y_true).sum() + self.smooth
    denominator = torch.pow(y_pred,2).sum() + torch.pow(y_true, 2).sum() + self.smooth
    dc = numerator / denominator
    loss = torch.mean(dc)
    return 1-loss

class BCE_Dice_Loss(torch.nn.Module):
  def __init__(self):
    super(BCE_Dice_Loss, self).__init__()
    self.bce_loss_func = torch.nn.BCEWithLogitsLoss()
    self.dice_loss_func = BinaryDiceLoss()

  def forward(self, y_pred, y_true):
    bce_loss  = self.bce_loss_func(y_pred, y_true)
    dice_loss = self.dice_loss_func(y_pred, y_true)
    return bce_loss + dice_loss

def set_loss_fn(opt):
    if opt.loss_fn == "binary_cross_entropy":
        return torch.nn.BCEWithLogitsLoss()
    elif opt.loss_fn == "dice":
        return BinaryDiceLoss()
    elif opt.loss_fn.lower() == "bce_dice_loss":
        return DiceCELoss(to_onehot_y=False, softmax=False, sigmoid=True)

####################################################
#--------------- VALIDATION MERICS ----------------#
####################################################

# Convert the image batch to a single batch after thresholding.
def get_array(image_batch):
  image_array = image_batch[0][0].cpu().detach().numpy()
  binary_image_array = np.where(image_array < 0.5, 0, 1)
  return binary_image_array

# Compute validation losses.
def compute_metrics(pred, gt, metrics, opt=None):
    dsc = dc(result=pred, reference=gt)                                                # Dice Coefficient.
    metrics["DSC"].append(dsc)
    return metrics

# This lr scheduler was taken from the this repository: https://github.com/cmpark0126/pytorch-polynomial-lr-decay
# The credit goes to its author Chunmyong Park (Username: cmpark0126).
class PolynomialLRDecay(_LRScheduler):
    """Polynomial learning rate decay until step reach to max_decay_step
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        total_epochs: after this step, we stop decreasing learning rate
        min_learning_rate: scheduler stoping learning rate decay, value of learning rate must be this value
        exp_power: The exp_power of the polynomial.
    """
    
    def __init__(self, optimizer, total_epochs, min_learning_rate=0.0, exp_power=0.9):
        if total_epochs <= 1.:
            raise ValueError('total_epochs should be greater than 1.')
        self.total_epochs = total_epochs
        self.min_learning_rate = min_learning_rate
        self.exp_power = exp_power
        self.last_step = 0
        super().__init__(optimizer)
        
    def get_lr(self):
        if self.last_step > self.total_epochs:
            return [self.min_learning_rate for _ in self.base_lrs]

        return [(base_lr - self.min_learning_rate) * 
                ((1 - self.last_step / self.total_epochs) ** (self.exp_power)) + 
                self.min_learning_rate for base_lr in self.base_lrs]
    
    def step(self, step=None):
        if step is None:
            step = self.last_step + 1
        self.last_step = step if step != 0 else 1
        if self.last_step <= self.total_epochs:
            decay_lrs = [(base_lr - self.min_learning_rate) * 
                         ((1 - self.last_step / self.total_epochs) ** (self.exp_power)) + 
                         self.min_learning_rate for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
                param_group['lr'] = lr

def keep_only_largest_connected_component(pred_array):
    """
    Takes a binary 3D tensor and removes all contours that
    include less voxels than the largest one.
    """
    # Compute connected components.
    seg = pred_array.astype(np.uint8)
    conn_comp = cc3d.connected_components(seg, connectivity=18)

    # Count number of voxels of each component and find largest component.
    unique, counts = np.unique(conn_comp, return_counts=True)

    try:
      # Remove largest component, because it is background.
      idx_largest = np.argmax(counts)
      val_largest = unique[idx_largest]

      counts = np.delete(counts, idx_largest)
      unique = np.delete(unique, idx_largest)

      idx_second_largest = np.argmax(counts)
      val_second_largest = unique[idx_second_largest]

      # Remove all smaller components.
      out = np.zeros_like(conn_comp)
      out = np.where(conn_comp == val_second_largest, 1, out)
      return out.astype(np.uint8)
    except:
      return seg


# Save the predicted image.
def save_val_volumes(pred_array=None, p_id=None, dataset_info=None, opt=None):

    predicted_array = pred_array.transpose(2,1,0)
    predicted_image = sitk.GetImageFromArray(predicted_array)

    # Set other image characteristics.
    predicted_image.SetOrigin(dataset_info["files"][p_id]["new_volume_info"]["origin"])
    predicted_image.SetSpacing(dataset_info["files"][p_id]["new_volume_info"]["spacing"])
    predicted_image.SetDirection(tuple(dataset_info["files"][p_id]["new_volume_info"]["direction"]))

    # Make the predictions folder is it doesn't exit.
    validation_predictions_folder = os.path.join(opt.results_path, "validation_predictions")
    if not os.path.exists(validation_predictions_folder):
        os.mkdir(validation_predictions_folder) 
    
    # Write the image to the disk.
    sitk.WriteImage(predicted_image, os.path.join(validation_predictions_folder, p_id  + ".seg.nrrd"))

# Save the predicted image after processing.
def save_val_post_processed(p_id=None, dataset_info=None, opt=None):

    # Get the original label.
    orig_label_name = dataset_info["files"][p_id]["orig_volume_info"]["labelFilePath"].split("/")[-1]
    orig_label_path = os.path.join(opt.data_root,
                                   dataset_info["files"][p_id]["orig_volume_info"]["labelFilePath"])
    orig_label_image = sitk.ReadImage(orig_label_path)
    orig_label_array = sitk.GetArrayFromImage(orig_label_image).astype(np.uint8).transpose(2,1,0)

    # Get the predictions folder.
    validation_predictions_folder = os.path.join(opt.results_path, "validation_predictions")

    # Get the saved validation prediction.
    saved_pred_path = os.path.join(validation_predictions_folder, p_id  + ".seg.nrrd")
    #saved_pred_image = sitk.ReadImage(saved_pred_path)

    # Resample the predicted image.
    resample_transform = tio.Resample(target = orig_label_path, label_interpolation = 'nearest')
    resampled_pred_image = resample_transform(tio.Image(saved_pred_path, type=tio.LABEL))

    # Delete the predicted image (image with new spacing).
    os.remove(saved_pred_path) 

    # Convert torch image to numpy.
    orig_predt_array = resampled_pred_image["data"][0].numpy().astype(np.uint8)
    
    # Get the largest connected component (non-background) from the predicted image.
    orig_predt_array = keep_only_largest_connected_component(orig_predt_array)

    # Get image from array. Arrgh! Don't forget to transpose. 
    predicted_image = sitk.GetImageFromArray(orig_predt_array.transpose(2,1,0))

    # Set other image characteristics.
    predicted_image.SetOrigin(orig_label_image.GetOrigin())
    predicted_image.SetSpacing(orig_label_image.GetSpacing())
    predicted_image.SetDirection(orig_label_image.GetDirection())

    # Make the predictions folder is it doesn't exit.
    validation_predictions_postprocessed = os.path.join(opt.results_path, "validation_predictions_postprocessed")
    if not os.path.exists(validation_predictions_postprocessed):
        os.mkdir(validation_predictions_postprocessed) 
    
    # Write the image to the disk.
    sitk.WriteImage(predicted_image, os.path.join(validation_predictions_postprocessed, orig_label_name))

    return orig_predt_array, orig_label_array, orig_label_name


def clean_up(opt=None):
    validation_predictions_folder = os.path.join(opt.results_path, "validation_predictions")

    try:
      os.rmdir(validation_predictions_folder)
    except OSError as error:
        print(error)
        print("Directory '%s' can not be removed" %validation_predictions_folder)