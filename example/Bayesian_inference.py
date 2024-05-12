import os
import sys
import numpy as np
import pandas as pd
import pickle as pk
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Union
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
sys.path.insert(0, '../src')
import LHS_DesignGenerator as lhs
import GPR
import HPCA as MyPCA
import Graph
import MCMC
import corner 
import HepSobol as SOB

"""

Module Overview:
    ----------------
This module provides tools for  Latin Hypercube Sampling (LHS), data processing, Gaussian Process Regression (GPR), Principal Component Analysis (PCA), 
sensitivity analyses, and Markov Chain Monte Carlo (MCMC) simulations.  

Latin Hypercube Sampling:

Generate samples from a multidimensional distribution using Latin Hypercube Sampling, 
facilitating efficient exploration of the parameter space.

Data Handling:
- Load, scale, and process experimental and model data.
- Handle missing or corrupted files gracefully.

MCMC Support:
- Define priors, generate samples, and visualize traces and distributions.
- Prepare prerequisites for MCMC simulations efficiently.

GPR and PCA:
- Construct GPR models, perform PCA for dimensionality reduction, and make model predictions.
- Validate GPR predictions against experimental or test data.

Sensitivity Analyses:
- Perform sensitivity analyses to understand the impact of model parameters on model outputs.
- Calculate Sobol indices to quantify the relative importance of input parameters.

Main Workflow:
- Orchestrate data processing, MCMC preparation, execution, and result visualization.
- Streamline the workflow for ease of use and readability.

Note: This module offers a comprehensive suite of tools for analyzing experimental and model data, 
      performing GPR, conducting MCMC simulations, sensitivity analyses, and visualizing results.
      This module serves as a comprehensive toolkit for researchers and practitioners engaged in data analysis, 
      modeling, and simulation in various scientific domains.
      For details of Bayesian analysis in relativistic heavy-ion collisions See:  (arXiv:1804.06469 [nucl-th])

please read README.txt before running the code
@Nachiketa Sarkar
"""

#####################################################################################################################################
#                      Load Needed experimental and Model Data Line: 21-175
#####################################################################################################################################
 

def load_exp_data(*file_names, file_dir=None):
    """
    Load experimental data from one or more files.

    Args:
        *file_names: Variable number of file names.
        file_dir (str): Directory containing the files.

    Returns:
        list: List of arrays containing experimental data.
    """
    exp_data_list = []
    
    # Determine the directory containing the files
    if file_dir is None:
        parent_dir = os.path.dirname(os.getcwd())+"/Input/"

    else:
        parent_dir = os.getcwd() + "/" + file_dir + "/"
     
    # Loop through each file name
    for file_name in file_names:
        try:
            file_path = parent_dir + file_name + ".txt"
            
            # Load experimental data from the file
            with open(file_path, 'r') as file:
                data = np.loadtxt(file_path, dtype=float)
                exp_data_list.append(data)
                
        except FileNotFoundError:
            print(f"Error: File '{file_name}' not found at path '{file_path}'.")
            
    return exp_data_list
    
    
def load_self_scaled_exp_data(*file_names, other_scale_factors, file_dir=None,plot=False):
    """
    Open and process all files passed as keyword arguments.

    Args:
        *file_names: Variable number of file names to be processed.
        other_scale_factors (list): List of scale factors corresponding to each file.
        file_dir (str): Directory path where the files are located. If None, defaults to the current working directory.

    Returns:
        exp_scale_data (ndarray): Scaled experimental data.
        exp_scale_error (ndarray): Scaled experimental error.
        exp_cov_mat (ndarray): Covariance matrix of the scaled experimental error.
        exp_sceled_factor (ndarray): Scaled factors used for scaling each data point.
    """
    
    # Lists to store scaled data, error, and factor for each file
    exp_sceled_data_list = []
    exp_sceled_error_list = []
    exp_sceled_factor_list = []
    
    # Set the directory path
    if file_dir is None:
        parent_dir = os.path.dirname(os.getcwd())+"/Input/"
    else:
        current_dir =  os.getcwd() + "/" + file_dir + "/"
        
    # Loop through each file and its corresponding scale factor
    for file_name, scale_factor in zip(file_names, other_scale_factors):
        try:
            # Construct the file path
            file_path = parent_dir + file_name + ".txt"
            
            # Open the file
            with open(file_path, 'r') as file:
                # Process the file here
                print(f"Processing {file_name}:")
                
                # Load data from the file
                data = np.loadtxt(file_path, dtype=float) 
                data_and_error = data[:, 1:3]   
                
                # Calculate scaled factor
                scaled_factor = data_and_error[:, 0] * scale_factor
                
                # Scale the data and error
                scaled_data_and_error = data_and_error / scaled_factor[:, np.newaxis]
                scaled_data = scaled_data_and_error[:, 0].reshape(1, -1)
                scaled_error = scaled_data_and_error[:, 1].reshape(1, -1)
                
                # Append scaled data, error, and factor to respective lists
                exp_sceled_data_list.append(scaled_data)  
                exp_sceled_error_list.append(scaled_error)  
                exp_sceled_factor_list.append(scaled_factor.reshape(1, -1))
                
                
        except FileNotFoundError:
            print(f"Error: File '{file_name}' not found at path '{file_path}'.")
    
    # Concatenate data, error, and factor lists to create final arrays
    exp_scale_data = np.concatenate(exp_sceled_data_list, axis=1)
    exp_scale_error = np.concatenate(exp_sceled_error_list, axis=1)
    exp_sceled_factor = np.concatenate(exp_sceled_factor_list, axis=1)
    
    # Calculate squared error for covariance matrix
    exp_scale_error = np.square(exp_scale_error)
    exp_cov_mat = np.diag(exp_scale_error.reshape(-1, ))
    
    # Plot data error matrix for visualization
    if plot:
      plt.figure(figsize=(8, 6))
      ax = sns.heatmap(exp_cov_mat, cmap='viridis', fmt='.2f')
      ax.set_title("Data error matrix Each block represents pT bin")
      plt.show()
    
    # Print scaled data
 
    
    return exp_scale_data, exp_scale_error, exp_cov_mat, exp_sceled_factor
 

def load_scaled_model_data(exp_scaled_factor, *file_names, file_dir=None):
    """
    Load and scale model data for training and testing.

    Args:
        exp_scaled_factor (ndarray): Scaled factor for experimental data.
        *file_names: Variable number of model file names to be processed.
        file_dir (str): Directory path where the files are located. If None, defaults to the current working directory.

    Returns:
        sceled_training_data (ndarray): Scaled training model data.
        testing_data (ndarray): Testing model data.
    """
    # Set the directory path
    if file_dir is None:
        parent_dir = os.path.dirname(os.getcwd())+"/Input/"
    else:
        current_dir = os.getcwd() + "/" + file_dir + "/"
    
    sceled_training_data_list = []
    testing_data_list = []
    
    # Loop through each file name
    for file_name in  file_names:
        try:
            file_path = parent_dir + file_name + ".txt"
            # Process training data
            with open(file_path, 'r') as file:
                data = np.loadtxt(file_path, dtype=float)
                sceled_training_data_list.append(data)
            # Process testing data
            file_path = file_path.replace("training", "testing")
            with open(file_path, 'r') as file:
                data = np.loadtxt(file_path, dtype=float)
                testing_data_list.append(data)
                
        except FileNotFoundError:
            print(f"Error: File '{file_name}' not found at path '{file_path}'.")
    
    # Concatenate training data and scale
    training_stack = np.hstack(sceled_training_data_list)
    sceled_training_data = training_stack / exp_scaled_factor
    
    # Concatenate testing data
    testing_data = np.hstack(testing_data_list)
    
    return sceled_training_data, testing_data
##############################################################################################
# User Define Function Needed in MCMC: Line no-191-250
############################################################################################
def model_systematic_error(sigma):
    """
    Calculate the systematic error in the model calculation.

    Args:
        sigma (float): Parameter representing the systematic error.

    Returns:
        float: The systematic error in the model calculation.
    """
    s = 0.05  # Scale parameter
    return sigma * sigma * np.exp(-sigma / s)

def Kernel():
    try:
        ptp= [0.090,0.5,0.5,0.8,0.05,2,3]
        ptp1 = [0.25, 0.13, 0.12, 0.234, 0.013]

        kernel = (
            1.5 * Matern(
                length_scale=ptp,
                length_scale_bounds=np.outer(ptp, (.005, 60000.1))
            ) +
            WhiteKernel(
                noise_level=.01 ** 0.5,
                noise_level_bounds=(.005 ** 2, 0.09)
            )
        )
        return kernel
    except Exception as e:
        logging.error(f"Error in Karnal function: {e}")
        raise

def log_prior(params,model_sigma=False):
    """
    Calculate the logarithm of the prior probability for given parameters.

    Args:
        params (array-like): Parameters for which to calculate the prior probability.

    Returns:
        float: The logarithm of the prior probability.
    """
    # Unpack parameters
    if model_sigma:
     T, BetaS, mpow, etabys, alpha, Rf, tf,m_sigma = params.T
    else:
     T, BetaS, mpow, etabys, alpha, Rf, tf = params.T
    # Check if parameters are within specified ranges
    if model_sigma:
     if (0.105 < T < 0.144 and
        0.72 < BetaS < 0.905 and
        0.71 < mpow < 0.975 and
        0.11 < etabys < 0.38 and
        0.05 < alpha < 0.133 and
        6.5 < Rf < 14.5 and
        6.5 < tf < 16.5 and
        0.001 < m_sigma < 0.4):
        return 0  # If within ranges, return 0 (logarithm of 1)
        
    else:       
     if (0.105 < T < 0.144 and
        0.72 < BetaS < 0.905 and
        0.71 < mpow < 0.975 and
        0.11 < etabys < 0.38 and
        0.05 < alpha < 0.133 and
        6.5 < Rf < 14.5 and
        6.5 < tf < 16.5 ):
        return 0  # If within ranges, return 0 (logarithm of 1)
    return -np.inf  # If not within ranges, return negative infinity (logarithm of 0)


def extra_opration(array):  
    '''
    Performs additional operation on the input array if needed.

    Parameters:
    array (numpy.ndarray): Input array.

    Returns:
    numpy.ndarray: Processed array.
    
    Notes:
    It's sometimes beneficial to first take the logarithm of the training data 
    to make the dataset closer to a normal distribution for better PCA performance.
    '''
    return np.exp(array)



def construct_input(parameter_list,num_pca_components, name, exp_data=None, exp_error=None, plot=False,pca_eigenvectors=None):
    """
    Prepare input data for MCMC and provide GPR predictions of model data using PCA. pca_eigenvectors

    Args:
        parameter_list (ndarray): Array containing sets of parameters for the GPR model.
                                  Each row represents a unique parameter set.
        num_pca_components(int) Number of pca components selected to train the GPR                          
        name (str): Identifier for the model, aiding in result tracking and analysis.
        exp_data (ndarray, optional): Array holding the experimental data.
                                       Each row represents an observation, and columns denote different features.
        exp_error (ndarray, optional): Array indicating the error associated with the experimental data.
                                        It matches the shape of exp_data, with each element representing the error
                                        for the corresponding observation.
        plot (bool, optional): Flag to trigger visualization of the experimental error matrix.
                               Set to True to generate a plot displaying the experimental error matrix.
        pca_eigenvectors: Full PAC gigen vector needed for calculation of model systematic error
    Returns:
        tuple: Tuple containing difference between model and experimental data (dy) and the total error matrix.

    Notes:
        - This function prepares input data required for Markov Chain Monte Carlo (MCMC) and provides GPR predictions
          of model data using Principal Component Analysis (PCA).
        - The 'parameter_list' argument should contain arrays of parameters for the GPR model, with each row representing
          a unique parameter set.
        - The 'name' argument provides an identifier for the model, aiding in result tracking and analysis.
        - 'exp_data' and 'exp_error' represent the experimental data and associated errors, respectively.
        - Setting 'plot' to True generates a visualization of the experimental error matrix, aiding in data analysis.
        - Ensure 'exp_data' and 'exp_error' are provided, as they are required arguments for accurate processing.
        - The function returns the difference between model and experimental data ('dy') and the total error matrix,
          which combines the model and experimental errors.
    """
 
    # Check if experimental data and error are provided
    if exp_data is None or exp_error is None:
        raise ValueError("Exp_Data and Exp_Error are required arguments.")
    
    if pca_eigenvectors is not None:
      # pca_eigenvectors is not given that means model systematic is not consider in the calculation
      # Split the array into for GPR list and model systematic value calculation
        model_sigma = parameter_list[:,parameter_list.shape[1]-1:]
        parameter_list = parameter_list[:,:-1]
 
    # Initialize GPR model
    gpr = GPR.GPR(kernel=Kernel, name=name, normalized=True)

    # Predict with GPR
    mean, std, cov = gpr.predict_gpr(parameter_list, return_covariance=True)

    # Perform PCA
    pca = MyPCA.HPCA(mean, num_pca_components, name)

    # Inverse transform PCA
    inverse_transformed_data, model_error_mat = pca.inverse_transform(pca_reduce_data=mean, gp_std=std)

    # Calculate difference between model and experimental data
    dy = inverse_transformed_data - exp_data

    # Combine experimental error and model error
    total_error_mat = exp_error + model_error_mat
    
    if pca_eigenvectors is not None:
       # pca_eigenvectors calculate model systematic error and add it to the total error
       model_systematic_error_matrix = np.diag(np.full(dy.shape[1], model_systematic_error(model_sigma)))
       model_systematic_mat=np.dot(np.dot(pca_eigenvectors, model_systematic_error_matrix), pca_eigenvectors.T)
 
       total_error_mat = exp_error + model_error_mat+model_systematic_mat
    # Plot the experimental error matrix if required
    if plot:
        np.linalg.cholesky(total_error_mat)
        Cov_inverse = np.linalg.inv(total_error_mat)
        plt.figure(figsize=(8, 6))
        ax=sns.heatmap(total_error_mat, cmap='viridis', fmt='.2f')
        ax.set_title("Data Plus Model error matrix Each Level represents pT bin")
        plt.show()

    return dy, total_error_mat




def model_prediction(training_model_results, name, scale_factor,model_systematic=False):
    """
    Predicts model data using Gaussian Process Regression (GPR) and scales the result.

    Args:
        training_model_results (array-like): Training data for the model.
        name (str): Name of the model.
        scale_factor (float): Scaling factor to apply to the predicted data.

    Returns:
        array-like: Scaled model predictions.
    """
    if model_systematic:
      training_model_results = training_model_results[:,:-1]


    # Initialize Gaussian Process Regression model
    gpr = GPR.GPR(kernel=Kernel, name=name, normalized=True)
    
    # Predict mean, standard deviation, and covariance using GPR
    mean, std = gpr.predict_gpr(training_model_results)
    
    # Initialize Principal Component Analysis model
    pca = MyPCA.HPCA(training_model_results, mean.shape[1], name)
    
    # Perform inverse transformation of reduced data using PCA
    rev_pca_data = pca.inverse_transform(pca_reduce_data=mean, plot=False)
    
    # Scale the transformed data by the scale factor
    scaled_data = rev_pca_data * scale_factor
    
    return scaled_data




def construct_prerequisites(training_model_results, training_parameter_list, test_results, test_parameter_list, name,num_pca_components,
                            scale_factor=None):
    """
    Prepares data and evaluates a Gaussian Process Regression (GPR) model.

    Args:
        training_model_results (numpy.ndarray): Training model results.
        training_parameter_list (numpy.ndarray): Training parameter list.
        test_results (numpy.ndarray): Test results.
        test_parameter_list (numpy.ndarray): Test parameter list.
        name (str): Name of the model.
        scaler (numpy.ndarray, optional): Scaling factor for predictions. Defaults to None.

    Returns:
        None
    """
    
    # Note: This function prepares data and evaluates a Gaussian Process Regression (GPR) model. It performs PCA for
    # dimensionality reduction, initializes and trains the GPR model, and evaluates its performance by comparing
    # predicted results with actual test results. Visual checks are performed to ensure the validity of the PCA
    # transformation and the effectiveness of the trained GPR model.

    # Important: To perform Markov Chain Monte Carlo (MCMC), this function must be executed once, as all necessary
    # information for MCMC will be saved. You just have to pass the same name to the MCMC object pointer.

    directory=os.path.dirname(os.getcwd())+"/src/cache_files/"
    # Initialize HPCA model
    pca = MyPCA.HPCA(training_model_results, num_pca_components, name)

    # Performed PCA on model_results
    pc_result = pca.pca_transform(plot=True, matrix_plot=True)

    # Visually check PCA
    pca.inverse_transform(pc_result, training_model_results, plot=True, check_positive_definite=True)

    # Construct the training list with Parameters set and corresponding reduce pc list for GPR
    training_set_pc = np.concatenate((training_parameter_list, pc_result), axis=1)

    # Initialize GPR model
    gpr = GPR.GPR(kernel=Kernel, num_parameters=training_parameter_list.shape[1], name=name, normalized=True)

    # Train the data
    if not os.path.exists(directory + "gpr_" + name + ".dump"):
        gpr.train_gpr(training_list=training_set_pc)

    # Check GPR by reverse transformation via PCA and compare Data
    mean, std, cov = gpr.predict_gpr(test_parameter_list, return_covariance=True)
    if mean.shape[0] == 1:
        reconstructed_data, Y_error = pca.inverse_transform(pca_reduce_data=mean, plot=True, gp_std=std,
                                                      check_positive_definite=True)
    else:
        reconstructed_data = pca.inverse_transform(pca_reduce_data=mean, plot=True, gp_std=std,
                                             check_positive_definite=True)
    if scale_factor is not None:
        reconstructed_data = reconstructed_data * scale_factor
    Graph.plot_gpr_comparison(model_prediction=test_results, GPR_prediction=reconstructed_data, error=np.diag(Y_error))

    

def main():

    # Define local variables
    name = "scaled_all"  # Name of the Project (Do't change once fixed, lots of files will save automatically and loaded accordingly)
    num_pca_components = 10  # Number of Principal components to take in the analysis
    parameters_name = [r'$T_{F}$', r'$\beta_{0}$', r'$m$', r'$\eta/s$', r'$\alpha_0$', r'$R_{F}$', r'$\tau$'] 
    model_systematic = True  # Whether model systematic error included or not

    #####################################################################################################################################
    # This part needs to run only once; subsequent executions do not require re-running everything up to the MCMC section (lines 209-306).
    #-----------------------------------------------------------------------------------------------------------------------------------
    # Load self-scaled experimental data and scale factor
    exp_scale_data, exp_scale_error, exp_cov_mat,scale_factor=load_self_scaled_exp_data("pi(0139)min_PBPB_2.76_0-5", "K(0492)min_PBPB_2.76_0-5", "pr(0938)min_PBPB_2.76_0-5", 
                                                                                       other_scale_factors=[1, 1, 0.01])
    # Load training data scaled according to the experimental scaling factor of the corresponding quantity. 
    # This scaling is not the normalization of the data. We aim for every experimental observable to be one,
    # to give each observable equal priority and make everything unit-less for comparing different types of observables simultaneously. 
    # Note that test results are not scaled, as they should not be. 
    # We need to compare test results and model or experimental results in their original form.

 
    training_model_results,test_results=load_scaled_model_data(scale_factor,"pi_pT_spectra_training_set_0_5", "k_pT_spectra_training_set_0_5", "pr_pT_spectra_training_set_0_5")
    # Load training and testing parameter lists
    test_parameter_list = np.loadtxt("../Input/testing_parameter_list_0-5.txt", dtype=float)
    training_parameter_list = np.loadtxt("../Input/training_parameter_list_0-5.txt", dtype=float)


    # Construct prerequisites for MCMC
    #construct_prerequisites(training_model_results=training_model_results, training_parameter_list=training_parameter_list, 
                            #test_results=test_results[0].reshape(1, -1), test_parameter_list=test_parameter_list[0].reshape(1, -1), 
                            #name=name, num_pca_components=num_pca_components, scale_factor=scale_factor)

    # Prepare input data for MCMC and check for numerical issues
    #for row in test_parameter_list:
         #construct_input(parameter_list=row.reshape(1, -1), num_pca_components=num_pca_components, name=name, 
                        #exp_data=exp_scale_data, exp_error=exp_cov_mat)
    # Note: Don't turn on pca_eigenvectors here,  trun it on  in the mcmc calse instant if we want to include 
    # model systematic error in your calculation 
    
    bounds=[[0.105, 0.144],[0.72, 0.905],[0.71, 0.975],[0.11, 0.38],[0.05, 0.133],[6.5, 14.5],[6.5, 16.5]]
    
    
    # Initialize HepSobol object and generate Sobol indices
    sob = SOB.HepSobol(num_vars=7,parameters_name=parameters_name,bounds=bounds,name=name,scale_factor=scale_factor)
    ST,S1,S2=sob.genereate_sobol_indices(model_prediction)
 
    # Plot Sobol indices  
    Graph.plot_sobol_indices(ST, S1, S2,parameters_name,21,sobol_index_type=['S2'])
     
    # Plot Sobol indices as heatmap
    Graph.plot_sobol_indices_matrix([S2],level_x=parameters_name,sobol_index_type=['S2'])
 

    
    
    
    #####################################################################################################################################      
    # Final Stapes MCMC Start from Here
    ##################################################################################################################################### 
     
    # Load experimental data (without scaling)
    list_exp_data = load_exp_data("pi(0139)min_PBPB_2.76_0-5", "K(0492)min_PBPB_2.76_0-5", "pr(0938)min_PBPB_2.76_0-5")
     
    mcmc = MCMC.MCMC(fun_construct_input=construct_input, fun_log_prior=log_prior,num_pca_components=num_pca_components, 
                    exp_data=exp_scale_data, exp_error_mat=exp_cov_mat, name=name,model_systematic=model_systematic)
    
    # Generate lhs
    generator = lhs.LHS_DesignGenerator(
        parameters_name=parameters_name,
        initial_values=[0.105, 0.72, 0.71, 0.11, 0.05, 6.5, 6.5],
        final_values=[0.144, 0.905, 0.975, 0.38, 0.133, 14.5, 16.5],
        name="r"
    )
    
    # Run MCMC
    flat_samples_burn_in, flat_samples_production,labels = mcmc.run_mcmc(LHS_DesignGenerator=generator, production_steps=500, no_walkers=200, burn_in_steps=200)
    
    # Plot MCMC traces
    Graph.plot_mcmc_trace(mcmc_chain=flat_samples_production, parameters_name=labels)
    
    # Plot experiment model comparison
    Graph.plot_experiment_model_comparison(list_exp_data=list_exp_data, flat_samples_burn_in=flat_samples_burn_in, 
                                          flat_samples_production=flat_samples_production, name=name, model_prediction=model_prediction, 
                                          scale_factor=scale_factor, no_sample=40, num_observables=len(list_exp_data),figsize=(10, 8),model_systematic=model_systematic)
    
    # Plot corner plot

    Graph.corner_plot(mcmc_chain=flat_samples_production, labels=labels)
  
    plt.show()
     

if __name__ == '__main__':
    main()

