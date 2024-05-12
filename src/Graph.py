import logging
import os
from pathlib import Path
import re
import sys
import subprocess
import numpy as np
import operator
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
import pickle as pk
from termcolor import colored
import numpy as np
import seaborn as sns
import io 
import corner 
import matplotlib.gridspec as gridspec         
from itertools import combinations
from matplotlib.ticker import FixedLocator


 
        
def plot_gpr_comparison(model_prediction, GPR_prediction, error=None):
    """
    
    Compare the model prediction with the GPR prediction.

    Parameters:
    - model_prediction: numpy array, the observed data from the model
    - GPR_prediction: numpy array, the predicted data from Gaussian Process Regression
    - error: numpy array, the standard deviation or error associated with the predictions

    Prints a DataFrame showing the observed, predicted, difference, and standard deviation.
    Plots the comparison between observed and predicted values with error bars.

    Returns:  plt
    
    """
      
    # Calculate the differences between observed and predicted values
    differences = model_prediction.reshape(-1, 1) - GPR_prediction.reshape(-1, 1)

    # Merge observed, predicted, difference, and standard deviation arrays
    merged_array = np.column_stack((model_prediction.reshape(-1, 1), GPR_prediction.reshape(-1, 1), differences.reshape(-1, 1), error.reshape(-1, 1)))

    # Create a DataFrame for easier visualization
    df = pd.DataFrame(merged_array, columns=['Model Prediction', 'GPR Prediction', 'Difference', 'Standard Deviation'])

    # Print the DataFrame
    print(df)
    
    # Plot the comparison
    fig, ax = plt.subplots()
    ax.errorbar(df['Model Prediction'], df['GPR Prediction'], xerr=df['Standard Deviation'], fmt="o", color='green', markersize=9, mfc='none')
    ax.plot(df['Model Prediction'], df['GPR Prediction'], color='black', markersize=9)
    ax.set_xlabel('GPR Prediction', fontsize=15)
    ax.set_ylabel('Model Prediction', fontsize=15)
    ax.tick_params(axis='both', labelsize=13)
    ax.set_title('GPR Comparison')
    plt.tight_layout()
    plt.show()
    return plt
    
def plot_pca_comparison(original_data, pca_data, error=None):
    """
    Plot the comparison between original data and PCA prediction.

    Parameters:
    - original_data: numpy array, the original observed data
    - pca_data: numpy array, the data after applying Principal Component Analysis (PCA)
    - error: numpy array, optional, the standard deviation or error associated with the predictions

    Prints a DataFrame showing the original data, PCA prediction, and difference.
    Plots the comparison between original data and PCA prediction with error bars if provided.

    Returns:
    None
    """

    # Calculate the differences between original data and PCA prediction
    differences = original_data.reshape(-1, 1) - pca_data.reshape(-1, 1)

    # Merge original data, PCA prediction, and difference arrays
    merged_array = np.column_stack((original_data.reshape(-1, 1), pca_data.reshape(-1, 1), differences.reshape(-1, 1)))

    # Create a DataFrame for easier visualization
    df = pd.DataFrame(merged_array, columns=['Original Data', 'PCA Prediction', 'Difference'])

    # Print the DataFrame
    print(df)

    # Plot the comparison
    fig, ax = plt.subplots()
    ax.errorbar(df['Original Data'], df['PCA Prediction'], xerr=np.abs(df['Difference']), fmt="o", color='green', markersize=9, mfc='none')
    ax.plot(df['Original Data'], df['PCA Prediction'], color='black', markersize=9)
    ax.set_xlabel('Original Data', fontsize=15)
    ax.set_ylabel('PCA Prediction', fontsize=15)
    ax.tick_params(axis='both', labelsize=13)

    if error is not None:
        ax.set_title('PCA Prediction with Error Bars')
    else:
        ax.set_title('PCA Comparison')

    plt.tight_layout()
    plt.show()
    return plt
    
     

 
def plot_pca_effect(before_pca_data, after_pca_data, eigen_vectors=None):
    """
    Create a matplotlib graph with two subplots arranged in a 2x1 grid.

    Args:
        before_pca_data (array-like): Data before PCA transformation.
        after_pca_data (array-like): Data after PCA transformation.
        eigen_vectors (array-like, optional): Eigenvectors for visualization.

    Returns:
        None
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot data before PCA on the first subplot
    ax1.scatter(before_pca_data[:, 0], before_pca_data[:, 1], color='blue')
    ax1.set_title("Before PCA")
    ax1.set_xlabel('Observable 1', fontsize=15)
    ax1.set_ylabel('Observable 2', fontsize=15)

    # Plot data after PCA on the second subplot
    ax2.scatter(after_pca_data[:, 0], after_pca_data[:, 1], color='red')
    ax2.set_title("After PCA")
    ax2.set_xlabel('PC 1', fontsize=15)
    ax2.set_ylabel('PC 2', fontsize=15)

    if eigen_vectors is not None:
        for i, vector in enumerate(eigen_vectors[:3]):
            ax2.arrow(0, 0, vector[0], vector[1], color=plt.cm.tab10(i), width=0.02, head_width=0.02, head_length=0.02, length_includes_head=True, label=f'Eigenvector {i+1}')

    plt.legend()
    plt.tight_layout()
    plt.show()
    return plt
    
 
    
    
    
def plot_pca_transformation_matrix(eigen_vectors, eigen_values, num_components):
    """
    Create a heatmap to visualize the principal components and their corresponding eigenvalues.

    Args:
        eigen_vectors (array-like): Matrix of eigenvectors.
        eigen_values (array-like): Array of eigenvalues.
        num_components (int): Number of principal components.

    Returns:
        None
    """
    eigen_vectors = eigen_vectors[:, :num_components]  # Select a subset of principal components
    print("eigen_vectors shape:", eigen_vectors.shape)

    plt.figure(figsize=(10, 8))
    sns.set(font_scale=0.8)
    heatmap = sns.heatmap(eigen_vectors, annot=True, cmap='viridis', fmt='.2f')

    # Add eigenvalue annotations
    for i, val in enumerate(eigen_values[:num_components]):
        heatmap.text(i + 0.5, -0.5, f'$\lambda$: {val:.4f}', ha='center', fontsize=10, color='red')

    # Set x-axis labels to indicate principal components
    plt.xticks(np.arange(eigen_vectors.shape[1]) + 0.5, [f'PC{i + 1}' for i in range(num_components)], fontsize=10)

    plt.xlabel('Principal Components', fontsize=12)  # Moved xlabel to indicate principal components
    plt.title('Principal Components and Eigenvalues Heatmap', fontsize=14, pad=20)  # Added pad to adjust title position
    plt.ylabel('Features', fontsize=12)
    plt.tight_layout()
    plt.show()
 
 

import os

def plot_pca_variance_explained(eigenvalues, name=None, save=False):
    """
    Plot the cumulative explained variance ratio for principal components.

    Args:
        eigenvalues (array-like): Eigenvalues of principal components.
        name (str, optional): Name of the plot. Default is None.
        save (bool, optional): Whether to save the plot. Default is False.

    Returns:
        None
    """
    # Create the figure
    plt.figure(figsize=(9, 5), dpi=100)

    # Generate plot name if save is enabled
    plot_name = None
    if save and name is not None:
        # Create directory if it doesn't exist
        if not os.path.exists("../Graph"):
            os.makedirs("../Graph")
        plot_name = f"../Graph/plot_pca_variance_explained_{name}.png"

    # Compute cumulative explained variance ratio
    explained_variance_ratio_ = eigenvalues / np.sum(eigenvalues)
    variance_exp_cumsum = np.cumsum(explained_variance_ratio_).round(2) * 100

    # Create x-axis values
    x = np.arange(1, len(variance_exp_cumsum) + 1)

    # Plot cumulative explained variance
    plt.plot(x, variance_exp_cumsum, '--o', color='r', markersize=5, label='Cumulative Variance Explained')

    # Plot horizontal lines at 95% and 98% explained variance
    plt.axhline(y=95, color='r', linestyle='--', label='95% Explained Variance')
    plt.axhline(y=98, color='b', linestyle='--', label='98% Explained Variance')

    # Add labels and legend
    plt.ylabel('Variance Explained (%)', fontsize=12)
    plt.xlabel('Number of Principal Components', fontsize=12)
    plt.legend()

    # Show grid
    plt.grid(True)
    plt.tight_layout()

    # Save plot if enabled
    if save:
        if name is None:
            raise ValueError("Name must be provided when save is True.")
        plt.savefig(plot_name, format="png", dpi=300)  # Save plot in high resolution (300 DPI)

    plt.show()


    
 
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
import seaborn as sns

def plot_mcmc_trace(mcmc_chain, parameters_name):
    """
    Visualize the trace and density of parameters sampled through Markov Chain Monte Carlo (MCMC).

    Args:
        sampled_params (ndarray): Sampled parameter values. Shape: (n_samples, n_parameters).
        param_names (list): Names of parameters, formatted in LaTeX.
    """
    n_params = len(parameters_name)
    fig, axs = plt.subplots(n_params, 2, figsize=(12, 6 * n_params))

    for i, parameters_name in enumerate(parameters_name):
        # Plot the trace
        axs[i, 0].plot(mcmc_chain[:, i])
        axs[i, 0].set_ylabel(f'{parameters_name}', fontsize=16)
        axs[i, 0].set_xlabel('Sample', fontsize=12)
        
        # Plot the density
        sns.histplot(mcmc_chain[:, i], kde=True, ax=axs[i, 1], color='skyblue', stat='density')
        axs[i, 1].set_ylabel('Density', fontsize=16)
        axs[i, 1].set_xlabel(f'{parameters_name}', fontsize=12)
        axs[i, 1].yaxis.tick_right()
    plt.suptitle('MCMC Trace Plot')   
    plt.show()



 


def plot_experiment_model_comparison(list_exp_data, flat_samples_burn_in,flat_samples_production, name, model_prediction, scale_factor, no_sample,num_observables, figsize=(10, 8),model_systematic=False):
    """
    Plot experimental data and model predictions for comparison.

    Args:
        experimental_data (list): List containing experimental data matrices.
        burn_in_samples (ndarray): Samples from the burn-in phase of the MCMC chains.
        production_samples (ndarray): Samples from the production phase of the MCMC chains.
        model_name (str): Name of the model.
        model_prediction_func (function): Function to compute model predictions.
        scale_factor (int): Scaling factor for the model predictions.
        num_samples (int): Number of samples.
        num_observables (int): Number of observables.
        figsize (tuple): Size of the figure.

    Returns:
        plt: Matplotlib plot object.
    """
 
    # Compute model predictions for burn-in  samples
    all_ymodel = [model_prediction(np.array([flat_samples_burn_in[i, :] for i in range(no_sample)]), name, scale_factor,model_systematic=model_systematic)]
    int_ymodel_list = np.split(np.vstack(all_ymodel), num_observables, axis=1)
    
    # Compute model predictions for production samples
    start_index = len(flat_samples_production) - no_sample
    end_index = len(flat_samples_production)
    final_ymodel_list = np.split(np.vstack([model_prediction(np.array([flat_samples_production[i, :] for i in range(start_index, end_index)]), name, scale_factor,model_systematic)]), num_observables, axis=1)

    # Create the figure and axes
    fig, axes = plt.subplots(num_observables, 2, figsize=figsize)
    
    # Plot experimental data and model predictions
    for i, matrix in enumerate(list_exp_data):
     row = i    # Calculate the row index
     for j in range(2):
      col=j
      if num_observables >1:
        ax = axes[row, col]
      else:
        ax=axes[j]
   
      # Plot experimental data   
      eb=ax.errorbar(matrix[:, 0], matrix[:, 1], yerr=matrix[:, 2], fmt='o', color='blue', label='Experimental Data')
      
      # Plot model predictions
      for index in range(no_sample):
            if j==0:
              yval=int_ymodel_list[i][index, :].reshape(-1, 1)

            else:
              yval=final_ymodel_list[i][index, :].reshape(-1, 1)
            
            mp=ax.plot(matrix[:, 0],yval , color='orange', alpha=0.5)  # Use alpha to adjust transparency
            ax.set_yscale('log')  # Set y-axis to logarithmic scale
            
            
            # Add text for particle type
            ax.set_ylabel(r'$\frac{1}{2 \pi p_T}  \frac{d^2N}{dy dp_T}$', fontsize=15)  # Y-axis label in LaTeX format

            ax.set_xlabel('$p_{T} $ (GeV/C)', fontsize=15)
            ax.legend([eb], ['Experimental Data'], loc='upper right')
            ax.legend(['Model Predication'])
            if i==0:
              ax.text(0.5, 0.1, r'$\pi$', fontsize=15, ha='center', va='bottom', transform=ax.transAxes)
            if i==1:
              ax.text(0.5, 0.1, r'$K$', fontsize=15, ha='center', va='bottom', transform=ax.transAxes)
            if i==2: 
             ax.text(0.5, 0.1, r'$P$', fontsize=15, ha='center', va='bottom', transform=ax.transAxes)
             
    # Adjust layout to prevent overlapping
    plt.tight_layout()
    plt.show()
    return plt
    
def corner_plot(mcmc_chain, labels):
    """
    Generate a corner plot from flat samples.

    Args:
        flat_samples (ndarray): Flat samples from the MCMC chains.
        labels (list): List of parameter labels.

    Returns:
        None
    """
    # Define contour plot kwargs
    contour_kwargs = {'colors': ['blue', 'green', 'red']}
    
        
    # Create corner plot
    fig = corner.corner(
        mcmc_chain,
        show_titles=True,
        bins=50,
        color='green',
        plot_datapoints=True,
        plot_density=True,
        use_math_text=True,
        labels =labels,
        contour_kwargs=contour_kwargs,
        quantiles=(0.16, 0.84, 0.95)
    )

   
    
    
def plot_2d_projection_lhs(df, no_points, parlist):
    """
    Plot a 2D projection of Latin Hypercube Samples.

    Args:
        df (DataFrame): DataFrame containing the Latin Hypercube Samples.
        no_points (int): Number of points.
        parlist (list): List of parameter names.

    Returns:
        None
    """
    # Create figure and gridspec
    fig = plt.figure(1, figsize=(8, 6))
    gs = gridspec.GridSpec(2, 2, height_ratios=[0.05, 0.2], width_ratios=[1, 0.2])
    gs.update(left=0.1, right=0.95, bottom=0.08, top=0.93, wspace=0.02, hspace=0.03)

    # Scatter plot for the main subplot
    ax1 = plt.subplot(gs[1, 0])
    plt1 = ax1.scatter(df[parlist[0]], df[parlist[1]], marker='o', s=60, color='g', alpha=1)
    ax1.set_xlabel("${}$".format(parlist[0]))
    ax1.set_ylabel("${}$".format(parlist[1]))

    # Vertical histogram subplot
    ax1v = plt.subplot(gs[1, 1])
    ax1v.hist(df[parlist[1]], bins=40, orientation='horizontal', color='r', edgecolor='w')
    ax1v.set_xticklabels([])
    ax1v.set_yticklabels([])

    # Horizontal histogram subplot
    ax1h = plt.subplot(gs[0, 0])
    ax1h.hist(df[parlist[0]], bins=40, orientation='vertical', color='b', edgecolor='w')
    ax1h.set_yticklabels([])
    ax1h.set_xticklabels([])

    # Show the plot
    plt.show()

    
    
def plot_sobol_indices(ST, S1, S2, labels, ith_observable, sobol_index_type, color='#2187bb', horizontal_line_width=0.25):
    """
    Plot confidence intervals for different Sobol indices.

    Args:
        ST (tuple): Tuple containing mean and confidence interval for total Sobol indices.
        S1 (tuple): Tuple containing mean and confidence interval for first-order Sobol indices.
        S2 (tuple): Tuple containing mean and confidence interval for second-order Sobol indices.
        labels (list): List of labels for the variables.
        ith_observable (int): Index of the observable to plot confidence intervals for.
        sobol_index_type (list): List of Sobol indices corresponding to different Sobol index types.
        color (str, optional): Color for the confidence interval lines and points. Defaults to '#2187bb'.
        horizontal_line_width (float, optional): Width of the horizontal lines representing the confidence intervals. Defaults to 0.25.
    """
    # Extract mean and confidence interval data for the specified observable
    confidence_intervals = [ST[ith_observable][:,1 ], S1[ith_observable][:, 1], S2[ith_observable][:, 1]]
    means = [ST[ith_observable][:, 0], S1[ith_observable][:, 0], S2[ith_observable][:, 0]]
    
    temp_mean=[]
    temp_conf=[]
    if 'ST' in sobol_index_type:
      temp_mean.append(means[0])
      temp_conf.append(confidence_intervals[0])
    if 'S1' in sobol_index_type:
      temp_mean.append(means[1])
      temp_conf.append(confidence_intervals[1])
    if 'S2' in sobol_index_type:
      temp_mean.append(means[2])
      temp_conf.append(confidence_intervals[2])
    
    means=temp_mean
    confidence_intervals=temp_conf
    # Generate labels for pairwise interactions in S2 indices
    labels_s2 = [f"{labels[i]}-{labels[j]}" for i, j in combinations(range(len(labels)), 2)]

    
    # Create subplots
    fig, axs = plt.subplots(1, len(sobol_index_type), figsize=(15, 5))
    
    # Loop over different Sobol indices
    for j, (mean, confidence_interval, index) in enumerate(zip(means, confidence_intervals, sobol_index_type)):
        x = np.arange(1, len(mean) + 1)
        if len(sobol_index_type) == 1:
           ax = axs
        else:
          ax = axs[j]
        
        # Plot confidence intervals and mean values
        for i, (m, ci) in enumerate(zip(mean, confidence_interval)):
            left = x[i] - horizontal_line_width / 2
            top = m - ci
            right = x[i] + horizontal_line_width / 2
            bottom = m + ci
            if i==0:
             ax.plot([x[i], x[i]], [top, bottom], color=color, label='Confidence Interval')
            else:
             ax.plot([x[i], x[i]], [top, bottom], color=color)
            ax.plot([left, right], [top, top], color=color)
            ax.plot([left, right], [bottom, bottom], color=color)
            if i==0:
             ax.plot(x[i], m, 'o', color='#f44336', label='Mean Value')
            else:
             ax.plot(x[i], m, 'o', color='#f44336')
        # Set title and x-axis ticks
        ax.set_title(f'{index}')
        ax.set_xticks(x)
        ax.xaxis.set_major_locator(FixedLocator(x))
        
        # Set x-axis labels based on the type of Sobol index
        if index == "S2":
            ax.set_xticklabels(labels_s2, fontsize=12)
        else:
            ax.set_xticklabels(labels, fontsize=12, rotation=45)
        
        # Increase y-axis tick label size
        ax.tick_params(axis='y', labelsize=12)
    
    # Add legend
    if len(sobol_index_type) > 1:
       axs[-1].legend(loc='upper right', fontsize=12)
    else:
      axs.legend(loc='upper right', fontsize=12)
    
    plt.show()

    
def plot_sobol_indices_matrix(sobol_index, level_x, sobol_index_type):
    """
    Plot Sobol indices as heatmaps.

    Args:
        sobol_index (list of numpy.ndarray): List of numpy arrays representing Sobol indices.
        level_x (list): List of labels for the variables.
        sobol_index_type (list): List of types of Sobol indices corresponding to each array.

    Raises:
        ValueError: If the length of sobol_index does not match the length of sobol_index_type.
    """
    temp=[]
    for lis_in in sobol_index:
      mean = [matrix[:, 0] for matrix in lis_in]
      mean=np.array(mean)
      temp.append(mean)
     
    
    sobol_index=temp
    
    # Check if the length of sobol_index matches the length of sobol_index_type
    if len(sobol_index) != len(sobol_index_type):
        raise ValueError("Length and sequence of sobol_index must match length of sobol_index_type")

    # Generate labels for pairwise interactions in S2 indices
    labels_s2 = [level_x[i] + level_x[j] for i, j in combinations(range(len(level_x)), 2)]

    # Create subplots
    fig, axs = plt.subplots(1, len(sobol_index_type), figsize=(15, 5))

    # Loop over each np_array and its corresponding sobol_index_type
    for i, np_array in enumerate(sobol_index):
        # Select the appropriate subplot based on the number of types
        if len(sobol_index_type) == 1:
            ax = axs
        else:
            ax = axs[i]

        # Plot the heatmap
        sns.heatmap(np_array, ax=ax, cmap='viridis', cbar_kws={'label': 'Sobol Index'},yticklabels=False)

        # Set title for the subplot
        ax.set_title(f'{sobol_index_type[i]}')

        # Set x-axis ticks and labels
        ax.set_xticks(np.arange(np_array.shape[1]) + 0.5)
        if sobol_index_type[i] == "S2":
            ax.set_xticklabels([f'{labels_s2[j]}' for j in range(np_array.shape[1])], fontsize=10, rotation=45)
        else:
            ax.set_xticklabels([f'{level_x[j]}' for j in range(np_array.shape[1])], fontsize=10, rotation=45)

        # Add text annotations for y-axis
        ax.text(-0.1, 16.5, r'$\pi$', fontsize=12, ha='center', rotation=90)
        ax.text(-0.1, 36.5, r'$K$', fontsize=12, ha='center', rotation=90)
        ax.text(-0.1, 60.5, r'$\Pr$', fontsize=12, ha='center', rotation=90)

        # Set y-axis label
        ax.set_ylabel(r'$\frac{1}{2 \pi p_T}  \frac{d^2N}{dy dp_T}$', fontsize=15, labelpad=20)

        # Set x-axis label
        ax.set_xlabel('Observable', fontsize=12)

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
