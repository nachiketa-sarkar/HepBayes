import os
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn import preprocessing
import pickle
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Union

class GPRTrainingError(Exception):
    """Custom exception class for errors during GPR training."""

class GPR:
    def __init__(self, kernel=None, num_parameters=None, name="default_name", normalized=True):
        """
          This class implements a Gaussian Process Regressor (GPR) model for predicting model quantities based on provided training data.
          It offers methods for training the GPR model, predicting model quantities, and optionally returning covariance matrices. 
          The GPR model is trained using the training data, and optional testing data can be used for validation. 
          The trained model and scaling factors are saved for future use. 

          Note: For detailes about GPR See; Ref: http://www.gaussianprocess.org
                                            source: https://scikit-learn.org/stable/modules/gaussian_process.html
        Args:
            kernel (function): The kernel function for the GPR model.
            num_parameters (int): The number of parameters.
            name (str): The name of the GPR model.
            normalized (bool): Whether normalized have to done or not 
        """
        self.kernel = kernel
        self.num_parameters = num_parameters
        self.name = name
        self.normalized = normalized
        self.directory=os.path.dirname(os.getcwd())+"/src/cache_files/"

    def gaussian_process(self, parameter_list, model_output):
        """
        Train Data by  a Gaussian Process Regressor (GPR) model.

        Args:
            parameter_list (numpy.ndarray): The input parameters.
            output (numpy.ndarray): The observed output.

        Returns:
            GaussianProcessRegressor: The trained GPR model.
        """
        try:
            kernel = self.kernel()
            gaussian_process = GaussianProcessRegressor(kernel=kernel, alpha=0.0, n_restarts_optimizer=4)
            return gaussian_process.fit(parameter_list, model_output)
        except Exception as e:
            logging.error(f"Error in gaussian_process method: {e}")
            raise GPRTrainingError(f"Error in gaussian_process method: {e}")

    def train_gpr(self, training_list, test_list=None):
     """
     Train the Gaussian Process Regressor (GPR) model.

     Args:
        training_list (numpy.ndarray): The data containing parameter values and observed quantities.
        test_list (numpy.ndarray, optional): The data for testing the trained model.

     Raises:
        ValueError: If the number of parameters is not positive.

     Notes:
        This method trains the GPR model with the provided training data. Optionally, it can also
        compare the model predictions with test data if the test_list is provided.
     """
     if self.num_parameters is not None and self.num_parameters <= 0:
        raise ValueError("Number of parameters must be positive.")
        
 
     print("\033[91m\u25b2 GPR training Process Started \u25bc\033[0m")

     try:
        # Split training data into parameters and observations
        parameters = training_list[:, :self.num_parameters]   # Parameter List
        observations = training_list[:, self.num_parameters:]  # Model Output List

        logging.info("Loaded data for GPR training.")
        
        # Determine the number of observables
        num_observables = training_list.shape[1] - self.num_parameters  
        observable_arrays = np.hsplit(observations, num_observables) 

        means = []
        stds = []
        gpr_models = []
        
        # Iterate over each observable
        for i, observable_array in enumerate(observable_arrays):
            # Calculate mean and standard deviation of the current observable array
            mean = np.mean(observable_array)
            std = np.std(observable_array)

            # Append mean and standard deviation to their respective lists
            means.append(mean)
            stds.append(std)

            # If normalization is enabled, scale the current observable array
            if self.normalized:
                scaler = preprocessing.StandardScaler().fit(observable_array)
                scaled_array = scaler.transform(observable_array)
            else:
                scaled_array = observable_array  # If not normalized, use the original array

            # Perform GPR processing with the scaled or original array
            gpr_models.append(self.gaussian_process(parameters, scaled_array))

            # Print the iteration count
            print("Iteration:", i+1)
         
        # Save GPR and Scale factor: Mean and Variance    
        save_dir = Path(self.directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        gpr_file = save_dir / f"gpr_{self.name}.dump"
        mean_file = save_dir / f"scale_mean_{self.name}.dump"
        std_file = save_dir / f"scale_std_{self.name}.dump"
        
        print("std_file",std_file)
        with open(gpr_file, "wb") as f:
            pickle.dump(gpr_models, f)

        with open(mean_file, "wb") as f:
            pickle.dump(means, f)

        with open(std_file, "wb") as f:
            pickle.dump(stds, f)

        # Optionally, compare GPR predictions with test data
        if test_list is not None:
            if self.normalized:
                compare_gpr(test_list, saved_gpr=gpr_models, saved_scale_mean=means, saved_scale_std=stds)  
            else:
                compare_gpr(test_list, saved_gpr=gpr_models)  
                
        print("\033[92m\u25b2 GPR training completed \u25bc\033[0m")
        logging.info("GPR training completed.")
        
     # Log and raise an error if any exception occurs during GPR training
     except Exception as e:
        logging.error(f"Error during GPR training: {e}")
        raise GPRTrainingError(f"Error during GPR training: {e}")
        
        

    def predict_gpr(self, prediction_set: np.ndarray, return_covariance: bool = False) -> Tuple[np.ndarray, np.ndarray, Union[List[np.ndarray], None]]:
      """
      Predict Model quantities using the Gaussian Process Regressor (GPR) model.

      Args:
        prediction_set (numpy.ndarray): Input data for prediction.
        covariance (bool): Whether to return the covariance matrix.

      Returns:
        Tuple containing predicted means, standard deviations, and covariance matrices (if requested).
      """
      # Lists to store predicted means, standard deviations, and covariances
      predicted_means = []
      predicted_std_devs = []
      covariances = []

      # Load saved GPR models and scaling factors
      saved_models_dir = self.directory
      if self.normalized:
        saved_scale_std = pickle.load(open(saved_models_dir + f"scale_std_{self.name}.dump", "rb"))
        saved_scale_mean = pickle.load(open(saved_models_dir + f"scale_mean_{self.name}.dump", "rb"))
      saved_gpr = pickle.load(open(saved_models_dir + f"gpr_{self.name}.dump", "rb"))

      # Iterate over each row of  prediction_set: np.ndarray
      for i, gpr_model in enumerate(saved_gpr):
        # Predict mean and covariance using the trained GPR model
        if return_covariance:
          mean_gpr, cov_gpr = gpr_model.predict(prediction_set, return_cov=True)
          std_gpr = np.sqrt(np.diag(cov_gpr))
        else:
          mean_gpr, std_gpr = gpr_model.predict(prediction_set,return_std=True)
          
        # Apply scaling if provided
        if saved_scale_mean is not None and saved_scale_std is not None:
            predicted_means.append(saved_scale_std[i] * mean_gpr + saved_scale_mean[i])
            predicted_std_devs.append(saved_scale_std[i] * std_gpr)
        else:
            predicted_means.append(mean_gpr)
            predicted_std_devs.append(std_gpr)

        # Append covariance matrix to the list if Covariance is True
        if return_covariance:
            covariances.append(cov_gpr)

      # Convert lists to numpy arrays
      predicted_means = np.array(predicted_means)
      predicted_std_devs = np.array(predicted_std_devs)
      if return_covariance:
        covariances = np.array(covariances)
 

      # Return covariance matrices if requested
      if return_covariance:
        return predicted_means.T, predicted_std_devs.T, covariances.T
      else:
        return predicted_means.T, predicted_std_devs.T
        
        
    def compare_gpr(self, test_list: np.ndarray, saved_gpr: List[GaussianProcessRegressor], 
               saved_scale_mean: Union[np.ndarray, None] = None, saved_scale_std: Union[np.ndarray, None] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compare GPR predictions with actual data.

        Args:
            data_list (numpy.ndarray): Data containing parameter values and observed quantities.
            saved_gpr (list): List of trained GPR models.
            saved_scale_mean (numpy.ndarray or None): Array of mean values for scaling.
            saved_scale_std (numpy.ndarray or None): Array of standard deviations for scaling.

        Returns:
            Tuple containing predicted means and standard deviations.
        """
        try:
            # Extract parameter values and observed quantities
            parameters = test_list[:, :self.num_parameters]
            observations = test_list[:, self.num_parameters:]

            # Perform GPR prediction
            means, stds = self.predict_gpr(parameters, saved_gpr=saved_gpr, saved_scale_mean=saved_scale_mean, saved_scale_std=saved_scale_std)

            # Calculate the difference between observed and predicted values
            differences = observations.reshape(-1, 1) - means.reshape(-1, 1)

            # Merge observed, predicted, difference, and standard deviation arrays
            merged_array = np.concatenate((observations.reshape(-1, 1), means.reshape(-1, 1), differences.reshape(-1, 1), stds.reshape(-1, 1)), axis=1)

            # Create a DataFrame for easier visualization
            df = pd.DataFrame(merged_array, columns=['Model Predication ', 'GPR Predication ', 'Difference', 'Standard Deviation'])

            # Print the DataFrame
            print(df)

            # Plot the comparison
            fig, ax = plt.subplots()
            ax.errorbar(df['Model Predication'], df['GPR Predication'], xerr=df['Standard Deviation'], fmt="o", color='green', markersize=9, mfc='none')
            ax.plot(df['Model Predication'], df['Model Predication'], color='black', markersize=9)
            ax.set_xlabel('Model Predication', fontsize=15)
            ax.set_ylabel('GPR Predication', fontsize=15)
            ax.tick_params(axis='both', labelsize=13)
            plt.tight_layout()
            plt.show()

 

        except Exception as e:
            logging.error(f"Error during comparison: {e}")
            raise GPRTrainingError(f"Error during comparison: {e}")

 
