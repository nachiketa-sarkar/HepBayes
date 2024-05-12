import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
import pickle as pk
from termcolor import colored
import Graph

class HPCA:
    def __init__(self, data, num_components, name,standardization=True,pca=None,eigenvectors=None,pca_reduce_data=None):
    
        """
        Initializes a PCAHandler object.

        Parameters:
        - data: Input data for PCA (numpy array or pandas DataFrame).
        - num_components: Number of principal components to retain.
        - name: Name or identifier for the instance.
        - standardization: Flag indicating whether to standardize the data (default: True).

       Purpose:
        The HPCA class provides a convenient interface for performing Principal Component Analysis (PCA)
        on input data. This method initializes the PCAHandler object with necessary parameters and attributes
        required for PCA computation and transformation.

       Notes:
       - 'data': Input data should be provided as a numpy array or pandas DataFrame.
       - 'num_components': Specifies the number of principal components to retain after PCA transformation.
       - 'name': A unique identifier for the HPCA instance, used for saving related files.
       - 'standardization': If set to True, the input data will be standardized before PCA transformation.
         Standardization is recommended when the features in the input data have different scales.
       - 'additional_operation': An optional function that can be passed to perform an extra operation on the
          reconstructed data during inverse transformation. This function should accept the reconstructed data
          as input and return the modified data.
        """
        self.data = data
        self.num_components = num_components
        self.name = name
        self.standardization = standardization

        self.scaler=preprocessing.StandardScaler()
        self.pca = pca
        self.full_pca = None
        self.eigenvectors = eigenvectors
        self.first_n_components = None
        self.pca_reduce_data = pca_reduce_data
        self.directory=os.path.dirname(os.getcwd())+"/src/cache_files/"

 
        

    def pca_transform(self, plot=False, matrix_plot=False):
      """
      Applies Principal Component Analysis (PCA) to the input data.

      Parameters:
      - plot: Flag indicating whether to plot the PCA effect (default: False).
      - matrix_plot: Flag indicating whether to plot the PCA matrix (default: False).

      Returns:
      - Transformed data.
      """
 
 
      # Standardize the data if required
      if self.standardization:
        scaler = preprocessing.StandardScaler().fit(self.data)
        scaled_data = scaler.transform(self.data)
        with open(self.directory + f"pca_scale_{self.name}.pkl", "wb") as scale_file:
            pk.dump(scaler, scale_file)
      else:
        scaled_data = self.data

      # Fit PCA model and transform data
      pca = PCA(copy=False, svd_solver='full')
      full_pca = pca.fit_transform(scaled_data)
    
      # Save eigenvalues and eigenvectors
      with open(self.directory + f"Full_EigenValue_{self.name}.pkl", "wb") as eval_file:
        pk.dump(pca.explained_variance_, eval_file)
      with open(self.directory + f"Full_EigenVector_{self.name}.pkl", "wb") as ev_file:
        pk.dump(pca.components_, ev_file)
    
      # Extract the first n components
      eigenvectors = pca.components_
      first_n_components = eigenvectors[:self.num_components, :].T
    
      # Reduce the data dimensionality
      pca_reduce_data = np.dot(scaled_data, first_n_components)
        
      # Plot PCA effect if required
      if plot:
        Graph.plot_pca_effect(scaled_data, pca_reduce_data)
    
      # Plot PCA matrix if required
      if matrix_plot:
        Graph.plot_pca_transformation_matrix(eigenvectors * scaler.scale_,
                              pca.explained_variance_ * scaler.scale_**2,
                              self.num_components)

      return pca_reduce_data
        
    def inverse_transform(self,pca_reduce_data,original_data=None,plot=False,extra_opration=None,gp_std=None,check_positive_definite=False):
        
        """
         Reconstructs original data from reduced dimensions.

          Parameters:
         - pca_reduce_data: Reduced dimensional data.
         - original_data: Original data for comparison (optional).
         - plot: Flag indicating whether to plot the reconstructed data (default: False).
         - extra_operation: Additional operation to apply on the reconstructed data (optional).
         - gp_std: Standard deviation for Gaussian process (optional).
         - check_positive_definite: Flag indicating whether to check for positive definiteness (default: False).

         Returns:
        - Reconstructed original data.
        """
 
        
        # Load scaler, eigenvectors, and eigenvalues 
        with open(self.directory + f"pca_scale_{self.name}.pkl", "rb") as scale_file:
             self.scaler = pk.load(scale_file)
        with open(self.directory + f"Full_EigenVector_{self.name}.pkl", "rb") as ev_file:
             self.eigenvectors = pk.load(ev_file)
        eigenvalue = pk.load(open(self.directory+"Full_EigenValue_"+ self.name+".pkl","rb")) 
        
        
        reconstructed_data = (np.dot(pca_reduce_data, self.eigenvectors[:self.num_components]))
        
 
        # Reconstruct data: 
        # The original data (X) can be reconstructed from the reduced data (Z) using the 
        # eigenvectors (V):  X=Z×V^T Here, Z is the reduced data obtained from PCA, and V^T represents the 
        # transpose of the PCA eigenvectors matrix.
        
        # Inverse standardization if required
        if self.standardization:
          reconstructed_data = self.scaler.inverse_transform(reconstructed_data)
        
        # Apply extra operation if provided
        if extra_opration is not None:
          reconstructed_data=extra_opration(reconstructed_data)
          
        # Calculate covariance matrix if gp_std is provided
       
        """
	   Mathematical Explanation:
	   1. Error Propagation: Propagate errors from reduced PCA space to the original space by adjusting the covariance matrix.
	   2. Calculation of Error Matrix:
   	      Error matrix in real space (Σ_y) is calculated using the formula: Σ_y = V Σ_z V^T, where V (eigenvectors) is the PCA transformation matrix.
	   3. Diagonal Covariance Matrix:
   	      The principal components are uncorrelated by construction, so the covariance matrix is diagonal.
   	      Σ_z = diag(σ_1^2 , . . . , σ_m^2, k_1^2 , . . . , k_n^2), where σ's are the GPR std in the reduced PCA space,
   	      and k_n^2 are the variances of the remaining neglected PCs.
	   4. Numerical Stability:
   	      To ensure numerical stability, a small term is added to the variances of the neglected PCs (k_n^2).
        """

        if gp_std is not None and reconstructed_data.shape[0] == 1:
        
         # Square the elements of gp_std
         gp_std=np.square(gp_std)
         
         # Reshape eigenvalues for compatibility
         eigenvalue=eigenvalue.reshape(1, -1)
         
         # Concatenate gp_std with 50 times scaled eigenvalues (starting from the num_components + 1-th component) to form the covariance matrix
         cov_mat = np.concatenate((gp_std.reshape(1,-1), 50.*eigenvalue[:,self.num_components: ]), axis=1)
         
         # Apply an additional operation if provided
         if extra_opration is not None:
            cov_mat=np.abs(reconstructed_data)*cov_mat
          
         # Convert the concatenated covariance matrix to a diagonal matrix  
         cov_mat = np.diag(cov_mat.reshape(-1,))
         
         # Optionally, scale the diagonal covariance matrix if standardization is enabled
         if self.standardization:           
          cov_mat=np.dot(np.diag(self.scaler.scale_), cov_mat)
          
         #computes the error covariance matrix (Σy) in the original space         
         Y_error=np.dot(np.dot(self.eigenvectors, cov_mat), self.eigenvectors.T)
                    
         # Optionally, check for positive definiteness of the covariance matrix
         #Should Off at MCMC to reduce time
         if check_positive_definite:
          np.linalg.cholesky(Y_error)    
          Cov_inverse=np.linalg.inv(Y_error)
        
        # Plot reconstructed data if required
        if plot:
          if original_data is not None:
            Graph.plot_pca_comparison(original_data, reconstructed_data)
          Graph.plot_pca_variance_explained(eigenvalue,save=True,name=self.name)
          
        if gp_std is not None and reconstructed_data.shape[0] == 1:
         return    reconstructed_data,Y_error
        else:
         return reconstructed_data

