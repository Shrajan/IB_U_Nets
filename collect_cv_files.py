import os, shutil, argparse, sys, pandas
import numpy as np

def copy_file(source=None, destination=None):
    try:
        shutil.copy2(source, destination)

    # If source and destination are same
    except shutil.SameFileError:
        os.remove(destination)
        shutil.copy2(source, destination)
    
    # If there is any permission issue
    except PermissionError:
        print("Permission denied.")
    
    # For other errors
    except:
        print("Error occurred while copying file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_path", type=str, default="results/test/", required=False, help="Path to read and store the results.")
    parser.add_argument('--k_fold', type=int, default=5, required=False, choices = [5,8], help='Choose between 5-fold cross validation or 8-fold cross validation. ')
    opt = parser.parse_args()

    main_exp_path = opt.exp_path
    k_fold = opt.k_fold
    
    cv_files_postprocessed = os.path.join(main_exp_path, "cv_files_postprocessed")

    # List to store all DSC values.
    all_DSCs = []

    if os.path.exists(main_exp_path):
        # Make the predictions folder is it doesn't exit.
        if not os.path.exists(cv_files_postprocessed):
            os.mkdir(cv_files_postprocessed)
    else:
        sys.exit("The experiment path doesn't exist.") 

    for fold in range(k_fold):

        # Collect the DSC values of each validtion sample.
        foldcsvFilePath = os.path.join(main_exp_path, "fold_"+str(fold), "val_results_detailed.csv")
        csvFile = pandas.read_csv(foldcsvFilePath)
        foldDSCs = csvFile["After process DSC"].tolist()
        all_DSCs += foldDSCs

        # Get the folder path of validation samples from each fold.
        fold_dir = os.path.join(main_exp_path, "fold_"+str(fold), "validation_predictions_postprocessed")
        
        # Get the names of the validation samples.
        try:
            fold_dir_files = os.listdir(fold_dir)
        except:
            print("{} doesn't exist.".format(fold_dir))
            sys.exit()

        # Copy each file in the source folder.
        for file in fold_dir_files:
            copy_file(source=os.path.join(fold_dir, file),
                      destination=os.path.join(cv_files_postprocessed, file))

    print("A total of {} files were collected.".format(len(all_DSCs)))
    meanDSC = round(np.mean(all_DSCs),3)
    medianDSC = round(np.median(all_DSCs),3)
    stdDSC = round(np.std(all_DSCs),3)
    minDSC = round(np.min(all_DSCs),3)
    maxDSC = round(np.max(all_DSCs),3)
    print("Mean DSC: {}, median DSC: {}, min DSC: {}, max DSC: {}, std DSC: {}."
            .format(meanDSC, medianDSC, minDSC, maxDSC, stdDSC))