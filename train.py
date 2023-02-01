# DOCUMENT INFORMATION
'''
    Project Name: IB U-Nets
    File Name   : train.py
    Code Author : Dejan Kostyszyn and Shrajan Bhandary
    Created on  : 14 March 2021
    Program Description:
        This program contains implementation of training program.
        * Saves the best model with highest mean validation DSC and complete model at the end.
        * Records training information such as losses, metrics etc.
            
    Versions:
    |----------------------------------------------------------------------------------------|
    |-----Last modified-----|----------Author----------|---------------Remarks---------------|
    |----------------------------------------------------------------------------------------|
    |    14 March 2021      |     Dejan Kostyszyn      |  Implemented necessary functions.   |
    |    18 April 2021      |     Shrajan Bhandary     |  Changed structure to object type.  |
    |    22 January 2022    |     Shrajan Bhandary     | Cleaned up stuff and added comments.|
    |----------------------------------------------------------------------------------------|
'''

# LIBRARY IMPORTS
import dataloader as custom_DL
import torch, os, time, csv, models, utils, tqdm
from options import Options
from torch.utils.data import DataLoader
import numpy as np
from monai.inferers import sliding_window_inference
import pandas as pd
# Faster runtime.
torch.backends.cudnn.benchmark = True

# IMPLEMENTATION

class Trainer():
    def __init__(self, opt=None):

        print("Initializing...")
        self.start_time = time.time()

        # Initializing.
        if opt is None:
            self.opt = Options().parse()
        else:
            self.opt = opt

        # Location to save the current training experiment.
        self.results_path = self.opt.results_path

        # Values to save best model.
        self.bestMeanDSC = 0.0 
        self.lowestLoss = 1000.0
        self.resume_train = self.opt.resume_train
            
        ####################################################
        #------------------ DATA LOADING ------------------#
        ####################################################

        self.train_dataset = custom_DL.Dataset(self.opt, split_type="train")
        self.val_dataset = custom_DL.Dataset(self.opt, split_type="val")

        train_dataset_idx, val_dataset_idx = self.train_dataset.get_split_idx(), self.val_dataset.get_split_idx()
        print("Training volumes: ", train_dataset_idx)
        print("Validation volumes: ", val_dataset_idx)
        
        self.trainloader = DataLoader(
            self.train_dataset,
            batch_size=self.opt.n_batches,
            shuffle=True,
            num_workers=self.opt.n_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
            prefetch_factor = self.opt.n_workers,
            )
        self.valloader = DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.opt.n_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor = self.opt.n_workers,
            )

        ####################################################
        #----------------- MODEL CREATION -----------------#
        ####################################################
        
        # Set the hardware device to run the model.
        device_id = 'cuda:' + str(self.opt.device_id)
        self.device = torch.device(device_id if torch.cuda.is_available() else 'cpu')

        # Get the model from the list of networks.
        self.model = models.get_model(opt=self.opt)

        # Move the model to the appropriate device.
        self.model.to(self.device)

        ####################################################
        #-------------- TRAINING PARAMETERS ---------------#
        ####################################################

        # Maximum training epochs.
        self.max_training_epochs = self.opt.training_epochs

        # Set optimizer
        self.optimizer = utils.set_optimizer(opt=self.opt, model_params=self.model.parameters())

        # Learning rate scheduler.
        self.lr_scheduler = utils.PolynomialLRDecay(optimizer=self.optimizer, total_epochs=self.max_training_epochs, min_learning_rate=0.0, exp_power=0.9)

        # Initialize loss functions.
        self.loss_func = utils.set_loss_fn(opt=self.opt)

    ###############################################################
    #------------------------- VALIDATION ------------------------#
    ###############################################################
    def val_step(self, save_volumes=False):
        """
        Validating the trained model on all validation samples.
        Returns:
            val_losses: list of all computed validation losses
            val_dsc: list of all computed validation DSCs
        """

        # List to store loss values and metrics generated.
        val_losses = []
        val_metrics = {
            "P_ID": [],
            "DSC": []
        }

        # Perform validation by setting mode = Eval
        self.model.eval()

        # Make sure that the gradients will not be altered or calculated.
        with torch.no_grad():

            # Consider a single batch at a given time from the complete dataset.
            for idx, (p_id, data, label) in enumerate(tqdm.tqdm(self.valloader, desc="Validating")):
                
                p_id = p_id[0]                                                                    # Get the string value of the patient ID.
                val_metrics["P_ID"].append(p_id)

                data, label = data.to(self.device), label.to(self.device)                         # Pass data and label to the device.
                with torch.cuda.amp.autocast():
                    out_M = sliding_window_inference(                                             # Feed the data into the model using sliding window technique.
                            inputs=data,
                            roi_size=tuple(self.opt.patch_shape),
                            sw_batch_size=4,
                            predictor=self.model,
                            overlap=0.25,
                            )
                    loss_M = self.loss_func(out_M, label.float())                                 # Calculate the loss function of the model prediction to the labelled ground truth.
                val_losses.append(loss_M.detach())                                                # Save the batch-wise validation loss.
                
                pred_array, label_array = utils.get_array(torch.sigmoid(out_M)), utils.get_array(label)
                val_metrics = utils.compute_metrics(pred=pred_array,                              # Calculate the performance metrices and save it in the dictionary.
                                                    gt=label_array, 
                                                    metrics=val_metrics, 
                                                    opt=self.opt)  
                
                if save_volumes:
                    utils.save_val_volumes(pred_array=pred_array,
                                           p_id=p_id,
                                           dataset_info=self.val_dataset.get_dataset_info(),
                                           opt=self.opt,
                    )
                
        return val_losses, val_metrics

    """
    This method trains the network.
    """
    def train(self, do_training = True, max_training_epochs=500, iterations_per_epoch = 250, latest_state_epoch=25, use_amp=True):
        """
        Trains the model for the given epochs from the starting point.
        Each epoch has by default 250 iterations (mini-batches).
        Uses automatic mixed precision when required.
        """
        ####################################################
        #---------------- LOAD STATE DICTS ----------------#
        ####################################################

        # Resume training or load states from previously completed model.
        if self.resume_train:
            found_saved_dicts = False
            found_history = False
            try:
                checkpoint = torch.load(os.path.join(self.results_path,'latest_net.sausage'), map_location=self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])              # Load the best weights.
                self.bestMeanDSC = checkpoint["val_dsc"]                                 # Highest mean validation DSC.
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])      # Load optimizer state.
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
                self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])   # Load scheduler state.

                found_saved_dicts = True
                print("Successfully loaded saved state dicts.")

                self.starting_epoch = checkpoint["epoch"]               

                if os.path.exists(self.results_path + "/history.csv"):
                    found_history = True
                    print("Old history file was found, so the new results will be appended to it.")    

            except:
                if found_saved_dicts is False:
                    print("Saved state dicts were not found or there was some issue. Training from scratch.")        

                if found_history is False:
                    print("History file was not found, so a new file will be created.")        
                    with open(self.results_path + "/history.csv", "w", newline="") as file:
                        writer = csv.writer(file, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        writer.writerow(["epoch","train_loss", "val_loss", "val_dsc"])   

        else:
            # Store epochs wise results.
            with open(self.results_path + "/history.csv", "w", newline="") as file:
                writer = csv.writer(file, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(["epoch","train_loss", "val_loss", "val_dsc"])
            self.starting_epoch = 1

        # For the first time when training starts.
        epoch = self.starting_epoch

        # List to store loss values per single batch.
        train_losses  = []

        # Print information
        print("\nEpoch {}/{}".format(epoch, max_training_epochs))
        print("*" * 30)
        epoch_start_time = time.time()
        train_start_time = time.time()

        # To count the number of steps/iterations.
        current_iteration = 0

        # Gradient scaling helps prevent gradients with small magnitudes from flushing 
        # to zero (“underflowing”) when training with mixed precision.
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        while do_training:
                    
            # Consider a single batch at a given time from the complete dataset.
            for idx, (p_id, data_patch, label_patch) in enumerate(self.trainloader):
                
                # Print training status.
                print("Iteration {}/{}".format(current_iteration+1, iterations_per_epoch), end="\r", flush=True)

                # Perform training by setting mode = Train
                self.model.train()

                self.optimizer.zero_grad()                                                            # Clears old gradients from the pevious step.
                data_patch, label_patch = data_patch.to(self.device), label_patch.to(self.device)     # Pass data and label to the device.
                
                with torch.cuda.amp.autocast():
                    out_M = self.model(data_patch)                                                    # Feed the data into the model.
                    loss_M = self.loss_func(out_M, label_patch.float())                               # Computes the loss of prediction with respect to the label.
                
                train_losses.append(loss_M)                                                           # Save the batch-wise training loss.
                scaler.scale(loss_M).backward()                                                       # Computes the derivative of the loss w.r.t. the parameters (or anything requiring gradients) using backpropagation.
                scaler.step(self.optimizer)                                                           # Use the optimizer to change the paramters based on their respective gradients. 
                scaler.update()                                                                       # Updates the scale for next iteration.
                current_iteration+=1                                                                  # Increment the number of iterations.

                # Check if epoch is done.
                if current_iteration % iterations_per_epoch == 0: 
                    train_end_time = time.time()
                    print("\nTraining with lr:{} was completed in {} s.".format(self.optimizer.param_groups[0]['lr'], train_end_time-train_start_time))

                    # Reset the current_iteration to 1.
                    current_iteration = 0

                    # Modify the learning rate.
                    self.lr_scheduler.step()

                    # Validation step.
                    val_losses, val_metrics = self.val_step(save_volumes=False)

                    ###############################################################
                    #---------------------- SAVING RESULTS -----------------------#
                    ###############################################################

                    result = dict()

                    # Training results
                    result['train_loss'] = torch.stack(train_losses).mean().item()
                    train_losses.clear()

                    # Validation results          
                    result['val_loss'] = torch.stack(val_losses).mean().item()
                    result['val_dsc'] = np.mean(val_metrics["DSC"])

                    # Save the current losses and metrics in the history file.
                    with open(self.results_path + "/history.csv", "a", newline="") as file:
                        writer = csv.writer(file, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        writer.writerow([ epoch, result["train_loss"], result["val_loss"], result["val_dsc"]])
                    
                    # Save the model that has the highest mean validation DSC.
                    epochMeanDSC = result['val_dsc']
                    if epochMeanDSC > self.bestMeanDSC: 
                        self.bestMeanDSC = epochMeanDSC
                        bestEpoch = epoch
                        torch.save({
                        "epoch": bestEpoch,
                        "val_dsc": self.bestMeanDSC,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.lr_scheduler.state_dict(),
                        "normalize": self.opt.normalize,
                        "patch_shape": self.opt.patch_shape,
                        "n_kernels": self.opt.n_kernels,
                        "clip": self.opt.clip,
                        "voxel_spacing": self.opt.voxel_spacing,
                        "input_channels": self.opt.input_channels,
                        "output_channels": self.opt.output_channels,
                        "no_shuffle": self.opt.no_shuffle,
                        "seed": self.opt.seed,
                        "model_name": self.opt.model_name
                        }, os.path.join(self.results_path, "best_net.sausage"))
                    
                    if epoch % latest_state_epoch == 0:
                        torch.save({
                        "epoch": epoch,
                        "val_dsc": epochMeanDSC,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.lr_scheduler.state_dict(),
                        "normalize": self.opt.normalize,
                        "patch_shape": self.opt.patch_shape,
                        "n_kernels": self.opt.n_kernels,
                        "clip": self.opt.clip,
                        "voxel_spacing": self.opt.voxel_spacing,
                        "input_channels": self.opt.input_channels,
                        "output_channels": self.opt.output_channels,
                        "no_shuffle": self.opt.no_shuffle,
                        "seed": self.opt.seed,
                        "model_name": self.opt.model_name
                        }, os.path.join(self.results_path, "latest_net.sausage"))

                    # Calculate the time required for each epoch.
                    epoch_end_time = time.time()
                    time_per_epoch = epoch_end_time - epoch_start_time

                    # Print the results after each epoch.
                    print("Finished epoch {} in {}s with results - train_loss: {:.4f}, val_loss: {:.4f}, val_dsc: {:.4f} \n".format(
                    epoch, int(time_per_epoch), result['train_loss'], result['val_loss'], result['val_dsc']))

                    # Increment the epoch count.
                    epoch+=1  
                
                    # End of training.
                    if epoch > max_training_epochs:
                        # Calculate the time required for full training.
                        do_training = False
                        
                        self.end_time = time.time()
                        print("\nCompleted training and validation in {}s".format(self.end_time - self.start_time))
                        break
                    
                    # Continue training.
                    else:
                        # Print information for next epoch
                        print("\nEpoch {}/{}".format(epoch, max_training_epochs))
                        print("*" * 30)
                        epoch_start_time = time.time() 
                        train_start_time = time.time()
            
            # End of training.
            if epoch > max_training_epochs:
                # Calculate the time required for full training.
                do_training = False
                return do_training
    
    """
    This method performs final inference.
    """
    def inference(self):
        """
        Using the best saved model, prediction is done on the validation samples.
        The prediction volumes are resampled to their respective original spacings, and 
        the largest connected component (non-background) is obtained. The post-processed volumes
        are then compared with the original ground-truth labels, and the results are saved.
        """
        print("\nPredicting with best saved model.")
        checkpoint = torch.load(os.path.join(self.results_path, 'best_net.sausage'),    # Load the saved checkpoint.
                                map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])                      # Load the best weights.
        val_losses, val_metrics = self.val_step(save_volumes=True)                      # Save the predicted samples.
        
        val_post_processed_metrics = {
            "P_ID": [],
            "DSC": []
        }

        with open(os.path.join(self.opt.results_path, "val_results_detailed.csv"), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["Original Name", "Sample ID", "Before process DSC", "After process DSC"])

        print("\nPostprocessing")
        for index, p_id in enumerate(val_metrics["P_ID"]):

            pred_array, label_array, label_name = utils.save_val_post_processed(p_id=p_id,    # Perform post-processing and return the arrays.
                                           dataset_info=self.val_dataset.get_dataset_info(),
                                           opt=self.opt,
                                           )
            val_post_processed_metrics = utils.compute_metrics(pred=pred_array,               # Calculate the performance metrices and save it in the dictionary.
                                               gt=label_array, 
                                               metrics=val_post_processed_metrics, 
                                               opt=self.opt)  
          
            # Display the results.
            print("DSC of patient volume {} before processing: {} and after processing: {}".
                    format(label_name, val_metrics["DSC"][index], val_post_processed_metrics["DSC"][-1]))

            # Write individual results into file.
            with open(os.path.join(self.opt.results_path, "val_results_detailed.csv"), 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([label_name, p_id, val_metrics["DSC"][index], val_post_processed_metrics["DSC"][-1]])
        
        # Write final results into file.
        with open(os.path.join(self.opt.results_path, "val_results.csv"), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["Stats", "Before process DSC", "After process DSC"])
            writer.writerow(["Mean", np.mean(val_metrics["DSC"]), np.mean(val_post_processed_metrics["DSC"])])
            writer.writerow(["Median", np.median(val_metrics["DSC"]), np.median(val_post_processed_metrics["DSC"])])
            writer.writerow(["Min", np.min(val_metrics["DSC"]), np.min(val_post_processed_metrics["DSC"])])
            writer.writerow(["Max", np.max(val_metrics["DSC"]), np.max(val_post_processed_metrics["DSC"])])
            writer.writerow(["Std", np.std(val_metrics["DSC"]), np.std(val_post_processed_metrics["DSC"])])

        # Print the final results.
        print("Overall DSC has mean: {}, median: {}, min: {}, max: {} and std: {}.".
              format(np.mean(val_post_processed_metrics["DSC"]), np.median(val_post_processed_metrics["DSC"]),
              np.min(val_post_processed_metrics["DSC"]), np.max(val_post_processed_metrics["DSC"]),
              np.std(val_post_processed_metrics["DSC"])))

        # Clean up.
        utils.clean_up(opt=self.opt)

if __name__ == "__main__":
    trainer = Trainer()
    do_training = True
    do_training = trainer.train(do_training = True, max_training_epochs=trainer.max_training_epochs, iterations_per_epoch = 250)
    if do_training is False:
        trainer.inference()
    print("\nThe experiment is done.")