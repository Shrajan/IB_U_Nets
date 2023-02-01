import numpy as np
import torch, argparse, os
import SimpleITK as sitk
import torchio as tio
import tqdm

def create_noisy_dataset(opt):
    # Create noise pipeline.
    if opt.noise == "random":
        noise = tio.transforms.RandomNoise(std=opt.random_std)
    elif opt.noise == "motion":
        noise = tio.transforms.RandomMotion(num_transforms=opt.motion_transforms)
    elif opt.noise == "blur":
        noise = tio.transforms.RandomBlur(std=opt.blur_std)
    
    patientFiles = os.listdir(opt.data_root)
    
    for patient in tqdm.tqdm(patientFiles, desc="Adding artifacts"):
        image_path = os.path.join(opt.data_root, patient)
        data = sitk.ReadImage(image_path)

        data_array = sitk.GetArrayFromImage(data).transpose(2, 1, 0).astype(np.float32) # Transpose, because sitk uses different coordinate system than pynrrd.
        data_array = torch.FloatTensor(data_array)

        # Add noise.
        data_array = noise(data_array.unsqueeze(0)).squeeze(0)

        # Copy new image data to old sitk image.
        result_image = sitk.GetImageFromArray(data_array.numpy().transpose(2, 1, 0))
        result_image.CopyInformation(data)

        # Write new image to folder
        fileWriter = sitk.ImageFileWriter()
        fileWriter.SetUseCompression(True)
        fileWriter.SetFileName(image_path)
        fileWriter.Execute(result_image)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_root", type=str, default="train_and_test", required=True, help="Path to data.")
  parser.add_argument("--noise", type=str, default="random", choices=["random", "motion", "blur"], required=True, help="Select noise to add.")
  parser.add_argument("--blur_std", type=float, default=2, required=False, help="The amount of standard deviation to be applied to create blur noise.")
  parser.add_argument("--random_std", type=float, default=45, required=False, help="The amount of standard deviation to be applied to create random noise.")
  parser.add_argument("--motion_transforms", type=int, default=5, required=False, help="The number of transforms to be applied to create motion noise.")
  opt = parser.parse_args()

  print("Started to create new noisy dataset...")
  create_noisy_dataset(opt)
