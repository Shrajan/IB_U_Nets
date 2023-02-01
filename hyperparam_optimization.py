# DOCUMENT INFORMATION
'''
    Project Name: Prostate Segmentation
    File Name   : hyperparam_optimization.py
    Code Author : Dejan Kostyszyn and Shrajan Bhandary
    Created on  : 12 May 2021
    Program Description:
        This program contains implementation of Hpbandster algorithm to retrieve the best 
        training parameters.
            
    Versions:
    |----------------------------------------------------------------------------------------|
    |-----Last modified-----|----------Author----------|---------------Remarks---------------|
    |----------------------------------------------------------------------------------------|
    |    12 May 2021        |     Dejan Kostyszyn      |  Implemented necessary functions.   |
    |    05 June 2021       |     Shrajan Bhandary     | Aligned train and validate methods. |
    |    22 January 2022    |     Shrajan Bhandary     | Cleaned up stuff and added comments.|
    |----------------------------------------------------------------------------------------|
'''

# LIBRARY IMPORTS
import numpy as np
import torch, csv, utils, os, copy, time
import torch.nn as nn
from monai.inferers import sliding_window_inference

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB as BOHB
from hpbandster.examples.commons import MyWorker
import hpbandster.core.result as hpres
import hpbandster.visualization as hpvis
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker
import hpbandster.core.result as hpres

import logging
logging.basicConfig(level=logging.WARNING)

from options import Options
from train import Trainer
import dataloader as DL

# IMPLEMENTATION

# Initializing.
opt = Options().parse()

class HyperOptTrainer(Trainer):
    def train_step(self, do_training = True, max_training_epochs=500, iterations_per_epoch = 250, 
                   use_amp=True, history_file="history.csv", overall_highest_dsc=0.0):

        # Create the history file.
        with open(history_file, "w", newline="") as file:
            writer = csv.writer(file, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["epoch","train_loss", "val_loss", "val_dsc"])

        # For the first time when training starts.
        epoch = 1

        # List to store loss values per single batch.
        train_losses  = []
        config_best_val_metrics = dict()
        config_best_mean_dsc = 0.0

        # To count the number of steps/iterations.
        current_iteration = 0

        # Gradient scaling helps prevent gradients with small magnitudes from flushing 
        # to zero (“underflowing”) when training with mixed precision.
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        while do_training:
                    
            # Consider a single batch at a given time from the complete dataset.
            for idx, (p_id, data_patch, label_patch) in enumerate(self.trainloader):
                
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
                    with open(history_file, "a", newline="") as file:
                        writer = csv.writer(file, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        writer.writerow([ epoch, result["train_loss"], result["val_loss"], result["val_dsc"]])
                    
                    # Save the model that has the highest mean validation DSC.
                    epochMeanDSC = result['val_dsc']
                    if epochMeanDSC > config_best_mean_dsc:
                        
                        # Only for this configuration.
                        config_best_mean_dsc = copy.copy(epochMeanDSC)
                        config_best_val_metrics = copy.deepcopy(val_metrics)

                        # Overall best network.
                        if epochMeanDSC > overall_highest_dsc: 
                            overall_highest_dsc = copy.copy(epochMeanDSC)
                            
                            bestEpoch = epoch
                            torch.save({
                            "epoch": bestEpoch,
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
                            }, os.path.join(self.results_path, "overall_best_net.sausage"))

                    # Print the status after each epoch.
                    print("Completed {}/{} epochs.".format(epoch, int(max_training_epochs)), end="\r", flush=True)

                    # Increment the epoch count.
                    epoch+=1  
                
                    # End of training.
                    if epoch > max_training_epochs:
                        do_training = False
                        break
                    
                    # Continue training.
                    else:
                        pass
            
            # End of training.
            if epoch > max_training_epochs:
                # Calculate the time required for full training.
                do_training = False
                return do_training, overall_highest_dsc, config_best_val_metrics

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
            for idx, (p_id, data, label) in enumerate(self.valloader):
                
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
                
        return val_losses, val_metrics


class MyWorker(Worker):
    def __init__(self, *args, sleep_interval=0, results_path=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.sleep_interval = sleep_interval
        self.config_number = 0
        self.results_path = results_path
        self.overall_highest_mean_dsc = 0.0

    def compute(self, config, budget, **kwargs):
        """
        Args:
            config: dictionary containing the sampled configurations by the optimizer
            budget: (float) amount of time/epochs/etc. the model can use to train
        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict)
        """

        # Saving hpbandster config in options.
        opt.lr = config["lr"]
        opt.beta1 = config["beta1"]
        opt.dropout_rate = config["dropout_rate"]
        opt.weight_decay = config["weight_decay"]
        opt.n_kernels = config["n_kernels"]

        self.config_number += 1

        print("\nSearch space #{}".format(self.config_number))
        print("Started worker with configuration:\n",
          "lr = {}, beta1 = {}, dropout rate = {}, weight_decay = {}, number of initial channels = {}".format(
            config["lr"], config["beta1"], config["dropout_rate"], config["weight_decay"], config["n_kernels"]
          )
        )


        ####################################################
        #----------------- CREATE TRAINER -----------------#
        ####################################################

        trainer = HyperOptTrainer(opt=opt)

        # Variable to determine when to store current best results.
        highest_mean_dsc = 0.0
        best_val_metrics = dict()

        history_file = self.results_path + "/config_" + str(self.config_number) + "_history.csv"  

        _, self.overall_highest_mean_dsc, best_val_metrics = trainer.train_step(do_training=True, 
                                                                max_training_epochs=budget, 
                                                                history_file=history_file,
                                                                overall_highest_dsc=self.overall_highest_mean_dsc)

        meanDSC, medDSC = np.mean(best_val_metrics["DSC"]), np.median(best_val_metrics["DSC"]) # Dice-soerensen coefficient.
        print("Best mean val DSC = {}, best median val DSC = {}".format(meanDSC, medDSC))

        accuracy = 1 - meanDSC # Using the inverse, because hpbandster minimizes.

        return {
            "loss": float(accuracy),  # Mean validation accuracy (1 - mean(DSC))
            "info": {
              'mean DSC': meanDSC,
              'median DSC': medDSC,
            }  # A bunch of metrics.
        }

    @staticmethod
    def get_configspace():
        config_space = CS.ConfigurationSpace()

        # Defining hyperparameters that shall be optimized.
        lr = CSH.UniformFloatHyperparameter('lr', lower=opt.lr_min, upper=opt.lr_max, default_value='1e-4', log=True)
        beta1 = CSH.UniformFloatHyperparameter('beta1', lower=opt.beta1_min, upper=opt.beta1_max, default_value=0.5, log=False)
        dropout_rate = CSH.UniformFloatHyperparameter('dropout_rate', lower=opt.dropout_rate_min, upper=opt.dropout_rate_max, default_value=0.25, log=False)
        weight_decay = CSH.UniformFloatHyperparameter('weight_decay', lower='1e-6', upper='1e-4', default_value='1e-5', log=True)
        n_kernels = CSH.UniformIntegerHyperparameter('n_kernels', lower=16, upper=48, default_value=32, log=False)
        
        config_space.add_hyperparameters([lr, beta1, dropout_rate, weight_decay, n_kernels])
        return config_space



def main():
    # Start a Nameserver.
    host = hpns.nic_name_to_host("lo")
    NS = hpns.NameServer(run_id=opt.run_id, host=host, port=opt.port)
    ns_host, ns_port = NS.start()

    # Start a worker.
    w = MyWorker(sleep_interval = 0, results_path=opt.results_path, host=host, nameserver=ns_host, nameserver_port=ns_port, run_id=opt.run_id)
    w.run(background=True)

    # Create a result logger for live result logging.
    result_logger = hpres.json_result_logger(directory=opt.results_path, overwrite=False)

    # Continue search from the last saved point.
    if opt.previous_search_path == "None":

        # Run an optimizer.
        bohb = BOHB(  configspace = w.get_configspace(),
                  run_id = opt.run_id,
                  host=host,
                  nameserver=ns_host,
                  nameserver_port=ns_port, result_logger=result_logger,
                  min_budget=opt.min_budget, max_budget=opt.max_budget,
              )

    else:
        if opt.previous_search_path == opt.results_path:
            raise Exception("Please use different path to store new results.")

        previous_run = hpres.logged_results_to_HBS_result(opt.previous_search_path)

        # Run an optimizer.
        bohb = BOHB(  configspace = w.get_configspace(),
                  run_id = opt.run_id,
                  host=host,
                  nameserver=ns_host,
                  nameserver_port=ns_port, result_logger=result_logger,
                  min_budget=opt.min_budget, max_budget=opt.max_budget,
                  previous_result = previous_run
               )
    res = bohb.run(n_iterations=opt.n_iterations)

    # Shut down worker and nameserver.
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()

    # Get all executed runs.
    all_runs = res.get_all_runs()

    # Analysis.
    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()

    print('\nBest found configuration:', id2config[incumbent]['config'])
    print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
    print('A total of %i runs where executed.' % len(res.get_all_runs()))
    print('Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in res.get_all_runs()])/opt.max_budget))

    final_msg = ""
    final_msg += 'Best found configuration:' + str(id2config[incumbent]['config']) + "\n"
    final_msg += 'A total of %i unique configurations where sampled.' % len(id2config.keys()) + "\n"
    final_msg += 'A total of %i runs where executed.' % len(res.get_all_runs()) + "\n"
    final_msg += 'Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in res.get_all_runs()])/opt.max_budget) +"\n"

    with open(opt.results_path + "/summary.txt", "w") as file_object:
      file_object.write(final_msg)
    
    # Visualization of runs.
    lcs = res.get_learning_curves()

    hpvis.interactive_HBS_plot(lcs, tool_tip_strings=hpvis.default_tool_tips(res, lcs))

if __name__ == "__main__":
    main()
