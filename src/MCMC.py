import emcee
import corner
import numpy as np
import random
import matplotlib.pyplot as plt
from IPython.display import display, Math
import pickle as pk
import scipy.linalg  
import matplotlib as mpl
import math 
import seaborn as sns
import corner 
import os
import sys

class MCMC:
    def __init__(self, fun_construct_input,fun_log_prior,num_pca_components, exp_data=None, exp_error_mat=None, name="default_name",model_systematic=False):
        """
        Initialize the MCMC model.

        Args:
            input_constructor (function): Function to construct input data. // User Define
            log_prior_func (function): Function to calculate the logarithm of the prior probability. // User Define
            num_pca_components(int) of pca components selected to train the GPR
            exp_data (array-like): Experimental data.
            exp_error_mat (array-like): Error matrix for the experimental data.
            name (str): The name of the MCMC model.
         
            # model_systematic is True if systematic error is to be added in the model calculation.
	    # The systematic error term in the model calculation is represented as σm^2 V V^T,
	    # where σm is the systematic error and V is a matrix.
	    # Since the "true" value of systematic error is unknown, it is left as a free parameter with a gamma distribution prior.
	    # The prior density function for sys is given by σm P(σ) ∝ σ^2 e^(-σ/s), where s = 0.05.
	    # This represents a gamma distribution with shape parameter k = 2 and scale parameter θ = s.

         
        MCMC Class Note:

        This class implements methods to perform Markov Chain Monte Carlo (MCMC) simulations.

        Usage:
        1. Initialize an instance of the MCMC class with the desired parameters.
        2. Use the `run_mcmc` method to run the MCMC simulation, specifying the production steps,
          number of walkers, burn-in steps, and other optional parameters.
        3. Optionally, provide a custom log probability function to evaluate the likelihood
         of the parameter space.

       Example: Foe More Details See Bayesian_inference 
         For details of MCMC see: https://emcee.readthedocs.io/en/stable/
         # Initialize an MCMC instance
         mcmc_instance = MCMC(name="example",construct_input=my_construct_input_function, log_probability=my_log_probability_function)

         # Run the MCMC simulation
         production_samples, burn_in_samples = mcmc_instance.run_mcmc(
         production_steps=1000,
         no_walkers=100,
         burn_in_steps=500,
         start_last_state=True
        )
        """
      # Analyze the obtained samples as needed

        self.exp_data = exp_data
        self.num_pca_components=num_pca_components
        self.exp_error_mat = exp_error_mat
        self.name = name
        self.fun_construct_input = fun_construct_input
        self.fun_log_prior = fun_log_prior
        self.count = 0  # Initialize global variable c
        self.model_systematic=model_systematic
        self.eigenvectors=None
        self.directory=os.path.dirname(os.getcwd())+"/src/cache_files/"
        if model_systematic: 
           # If model_systematic is True, load the full set of pca eigen vectors from a cached file.
           # The cached file is located in the directory "/cache_files/" and is named "Full_EigenVector_{self.name}.pkl",
           # where {self.name} represents the name of the model.

           with open(self.directory + f"Full_EigenVector_{self.name}.pkl", "rb") as ev_file:
             self.eigenvectors = pk.load(ev_file)
             

     
    def construct_input(self, parameter_list):
        """
        Construct input data.

        Args:
            parameter_list (array-like): List of parameters.
            Standeration (bool): Whether to standardize the data.

        Returns:
            array-like: Constructed input data. which is user define function
        """
        return self.fun_construct_input(parameter_list,self.num_pca_components,self.name, exp_data=self.exp_data, exp_error=self.exp_error_mat,pca_eigenvectors=self.eigenvectors)
 
    def log_prior(self, params):
        """
        Calculate the logarithm of the prior probability for given parameters.

        Args:
            params (array-like): Parameters for which to calculate the prior probability.

        Returns:
            float: The logarithm of the prior probability.
        """
        return self.fun_log_prior(params,self.model_systematic)

     
    
    def log_likelihood(self,params):
        """
         Calculate the logarithm of the likelihood function for given parameters.

         Args:
            params (array-like): Parameters for which to calculate the likelihood.

         Returns:
            float: The logarithm of the likelihood.
        """
        # Reshape the parameters into a 2D array
        params=params.reshape(1, -1) 
         
        # Construct input data and error matrix
        dy,error_mat=self.construct_input(params)
        
        # Calculate the logarithm of the likelihood using the multivariate normal log likelihood function
        return   self.mvn_loglike(dy.flatten().reshape(dy.size,1), error_mat) 
     
     
    def log_probability(self,params):
      """
       Calculate the log probability of the given parameters.

       Args:
        params: The parameters for which the log probability is to be calculated.

       Returns:
        float: The log probability of the parameters.
      """
     # Increment count
      self.count += 1  # Increment count=c+1
     # Calculate the log prior
      lprob = self.log_prior(params)
     
     # Check if log prior is finite
      if not np.isfinite(lprob):
        return -np.inf
        
     # Calculate the log likelihood
      lprob= self.log_likelihood(params) 
      if self.count % 100 == 0:
        print(f"\033[91m{'*' * 80}\n{self.count}th Steps with Probability {lprob} With Parameters {params} \n{'*' * 80}\033[0m")
      return lprob 
  
    def mvn_loglike(self, y, cov):
      """
      Calculate the log likelihood of the given data under a multivariate normal distribution.

      Args:
        y: The observed data.
        cov: The covariance matrix.

      Returns:
        float: The log likelihood of the data under the multivariate normal distribution.
      """
      # Set printing options
      np.set_printoptions(threshold=np.inf)
      np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})

      # Compute the Cholesky decomposition of the covariance matrix
      L = np.linalg.cholesky(cov)

      # Compute the inverse of the covariance matrix
      cov_inverse = np.linalg.inv(cov)

      # Compute the dot product of the inverse of the covariance matrix and the observed data
      alpha = np.dot(cov_inverse, y)

      # Compute the log likelihood
      log_likelihood = (-0.5 * np.dot(y.T, alpha) - np.log(L.diagonal()).sum()).item()

      return log_likelihood
     
  
    def run_mcmc(self,LHS_DesignGenerator, production_steps=None, no_walkers=None, burn_in_steps=None, discard=1, start_last_state=False,burn_in_only=False):
        """
        Run the MCMC simulation.

        Args:
        LHS_DesignGenerator: instant to LHS_DesignGenerator class
        production_steps (int): Number of steps for the production phase.
        no_walkers (int): Number of   no_walkers for burn-in.
        steps (int): Number of steps for each phase of burn-in.
        discard (int): Number of steps to discard.
        start_last_state (bool): Whether to start from the last state of the chain if available.
        LHS_DesignGenerator: instant of the class LHS_DesignGenerator
        Returns:
        array-like: Flattened samples from the MCMC chain.
        """
        
        if self.model_systematic:
        
         # If systematic error is considered in the model.
	 # In this case, the parameter name '$\sigma^y_m$' is appended to the list of parameter names in the LHS_DesignGenerator object.
	 # This parameter represents the systematic error scale factor.
	 # Additionally, initial and final values for this parameter are appended to their respective lists in the LHS_DesignGenerator object.
	 # The initial value is set to 0.001, and the final value is set to 0.04.
	 # These values represent the lower and upper bounds, respectively, for the systematic error scale factor during the Latin Hypercube Sampling (LHS) procedure.

         LHS_DesignGenerator.parameters_name.append(r'$\sigma^y_m$')
         LHS_DesignGenerator.initial_values.append(0.001)
         LHS_DesignGenerator.final_values.append(0.04)
        
        
        
        file_name_production = self.directory+f"mcmc_{self.name}_production.pkl"
        file_name_burn_in = self.directory+ f"mcmc_{self.name}_burn_in.pkl"
        
        sampler_production = None
        sampler_burn_in = None
        flat_samples_burn_in = None

        #  if burn_in_only Not True, Attempt to load the production sampler from the file if it exists 
        if not burn_in_only:
          try:
            if os.path.exists(file_name_production):
                with open(file_name_production, 'rb') as file:
                    sampler_production = pk.load(file)
                    
                # Load the burn-in sampler
                try:
                  with open(file_name_burn_in, 'rb') as file_burn_in:
                    sampler_burn_in = pk.load(file_burn_in)
                  flat_samples_burn_in = sampler_burn_in.get_chain(discard=1, thin=1, flat=True)
                except FileNotFoundError:
                       print(f"\033[91m {'*'*80}\n Error: The file '{file_name_burn_in}' was not found. Please run_mcmc(burn_in_only=True) \n{'*'*80}\033[0m")
                       sys.exit(1)
                # If starting from the last state, load the burn-in chain as well
                if start_last_state:
                         # Inform user about loading the chain from file and continue from the last state
                        print(f"\033[92m{'*'*50}\nMCMC Chain Found, Load Chain From File \"{self.name}\":\n{'*'*50}\033[0m")
                        input("\033[91mTo Continue From The Last Chain Point, Press Enter...\033[0m")
                        #load last save production chain
                        last_state = sampler_production.get_last_sample()
                        pos = last_state.coords
                        sampler_production.run_mcmc(pos, production_steps, progress=True, tune=True)
                        
                        # Save the Production sampler to the file
                        with open(file_name_production, "wb") as file:
                           pk.dump(sampler_production, file)
                           
                        # Inform user about completion of MCMC process
                        print(f"\033[92m{'*'*65}\nMCMC Process Complete, With: {production_steps} Production Steps and {pos.shape[0]} Walkers 😊!!\n{'*'*65}\033[0m")
                        return flat_samples_burn_in,sampler_production.get_chain(discard=discard, thin=1, flat=True),LHS_DesignGenerator.parameters_name

                else:
                        # If not starting from the last state, just return the production chain
                        print(f"\033[92m{'+'*65}\nMCMC Chain Found, Return Chain From File \"{self.name}\" 😊!!\n{'*'*65}\033[0m")
                        return flat_samples_burn_in, sampler_production.get_chain(discard=discard, thin=1, flat=True),LHS_DesignGenerator.parameters_name

          except FileNotFoundError:
            pass

        # If production sampler is not loaded or starting from the last state is not requested
        # Attempt to load the burn-in sampler if it exists and then proceeds to production steps
        if not sampler_production and not burn_in_only:
            # Attempt to load the burn-in sampler if it exists
            if os.path.exists(file_name_burn_in):
                try:
                    with open(file_name_burn_in, 'rb') as file:
                        sampler_burn_in = pk.load(file)
                        input("\033[92mBurn in Chain Found. To Continue Production Process, Press Enter...\033[0m")
                        last_state = sampler_burn_in.get_last_sample()
                        pos = last_state.coords
                        nwalkers, ndim = pos.shape
                        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability)
                        sampler.run_mcmc(pos, production_steps, progress=True, tune=True)
                        
                        # Save the Production sampler to the file
                        
                        with open(file_name_production, "wb") as file:
                           pk.dump(sampler, file)
                           
                        # Inform user about completion of MCMC process
                        print(f"\033[92m{'*'*65}\nMCMC Process Complete, With: {production_steps} Production Steps and {pos.shape[0]} Walkers 😊!!\n{'*'*65}\033[0m")
                        return sampler_burn_in.get_chain(discard=discard, thin=1, flat=True), sampler.get_chain(discard=discard, thin=1, flat=True),LHS_DesignGenerator.parameters_name

                except FileNotFoundError:
                    pass
        # If neither production nor burn-in sampler is found, continue with the normal MCMC process
        # If sampler is not loaded or starting from the last state is not requested
        # Prompt user if no saved MCMC chain is found and to continue from burn-in steps
        
        
        input("\033[91mNo Saved MCMC Chain Found. To Continue From Burn-in Steps Press Enter...\033[0m")
        
        # The burn-in process is split into two phases, each with half of the given burn-in steps, 
        # allowing for quicker convergence to the steady state.
        
        no_burn_step = burn_in_steps // 2
        
        pos = LHS_DesignGenerator.create_dataframe(no_walkers)

        nwalkers, ndim = pos.shape
        # --- Phase 1 of Burn-in Process ---
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability)
        print(f"\033[92m{'*'*65}\n1st Phase of Burn-in Process Start, With: {no_burn_step} Production Steps and {pos.shape[0]} Walkers 😊!!\n{'*'*65}\033[0m")
        sampler.run_mcmc(pos, no_burn_step, progress=True, store=True)
        
        with open(file_name_burn_in, "wb") as file:
           pk.dump(sampler, file)
        
        # Get the last positions from the burn-in phase
        B1 = sampler.flatchain[np.unique(sampler.flatlnprobability, return_index=True)[1][-nwalkers:]]
        sampler.reset()
        
        # --- Phase 2 of Burn-in Process ---
        print(f"\033[92m{'*'*65}\n2nd Phase of Burn-in Process Start, With: {burn_in_steps-no_burn_step} Production Steps and {B1.shape[0]} Walkers!!\n{'*'*65}\033[0m")
        B2 = sampler.run_mcmc(B1, burn_in_steps-no_burn_step, store=False, tune=True, progress=True)[0]
        sampler.reset()
        
        if not burn_in_only:
         # --- Production Phase of MCMC ---
         print(f"\033[92m{'*'*65}\n Production Phase of  MCMC Started, With: {production_steps} Production Steps and {B2.shape[0]} Walkers 😊!!\n{'*'*65}\033[0m")
         sampler.run_mcmc(B2, production_steps, progress=True, tune=True)
         # Save the Production sampler to the file
         with open(file_name_production, "wb") as file:
           pk.dump(sampler, file)

        sampler_p = pk.load(open(file_name_production, 'rb'))
        flat_samples_p = sampler_p.get_chain(discard=1, thin=1, flat=True)
        
        sampler_b = pk.load(open(file_name_burn_in, 'rb'))
        flat_samples_b = sampler_b.get_chain(discard=1, thin=1, flat=True)
        
        return flat_samples_b,flat_samples_p,LHS_DesignGenerator.parameters_name
     
 
 
