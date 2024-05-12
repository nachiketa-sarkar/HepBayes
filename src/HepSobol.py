import os
import numpy as np
import pandas as pd
from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
from SALib import ProblemSpec
import GPR
import pickle as pk
 

 
 
from pathlib import Path
''' 
Sensitivity analysis explores the effects of input parameter variations on a model or system's output. 
In the Sobol method, sensitivity indices quantify parameter sensitivity. Here's a concise breakdown of these indices:

First-order indices: Measure individual parameter impact on output variance, indicating their relative importance.

Second-order indices: Assess pairwise parameter interactions' influence on output variance.

Total-order index: Captures a parameter's overall impact on output variance, including direct and higher-order interaction effects.

For more details See: https://salib.readthedocs.io/en/latest/user_guide/getting-started.html
                      https://doi.org/10.1016/S0378-4754(00)00270-6.
''' 
 
class HepSobol:
  """
    Class for performing sensitivity analysis using Sobol indices in the HEP model.

    Attributes:
        num_vars (int): Number of parameters.
        parameters_name (list): Names of parameters.
        bounds (list): Parameter bounds.
        name (str): Name of the model.
        no_sample_point (int): Number of sample points (default is 2000).
        scale_factor (float): Scale factor for model prediction.
        directory (str): Directory for saving cache files.
        sp (ProblemSpec): ProblemSpec object for defining the problem.
  """

  def __init__(self, num_vars, parameters_name, bounds,name,scale_factor,no_sample_point=2000,print_on=False):
        """
        Initializes the HepSobol object with specified parameters.

        Args:
            num_vars (int): Number of parameters.
            parameters_name (list): Names of parameters.
            bounds (list): Parameter bounds.
            name (str): Name of the model.
            scale_factor (float): Scale factor for model prediction.
            no_sample_point (int, optional): Number of sample points (default is 2000).
        """

        self.num_vars = num_vars
        self.parameters_name = parameters_name
        self.bounds = bounds
        self.name=name
        self.no_sample_point=no_sample_point
        self.scale_factor=scale_factor
        self.directory=os.path.dirname(os.getcwd())+"/src/cache_files/"
        self.print_on=print_on
        self.sp=ProblemSpec({
    'num_vars': self.num_vars,
    'names': self.parameters_name,
    'bounds': self.bounds,
              'outputs': ['Y']        
})

 
 

  def genereate_sobol_indices(self,model_prediction):
      """
      Generate Sobol indices for sensitivity analysis.

      Args:
      model_prediction (function): Function for model prediction.

      Returns:
      tuple: Tuple containing total_ST_list, total_S1_list, and total_S2_list.
      """
      # Define file path for saving Sobol indices
      save_dir = Path(self.directory)
      sobol_file = save_dir / f"sobol_{self.name}.dump"
      
      # Generate Sobol parameters
      sobol_parameters = self.sp.sample_sobol(self.no_sample_point, calc_second_order=True)
      sobol_parameters=self.sp.samples
      
      # If Sobol indices file does not exist, calculate and save indices  
      if not os.path.exists(sobol_file):
         print(f"\033[91m {'*'*50}\n 		Generating Sobol Indices  \n{'*'*50}\033[0m")  
         # Calculate model prediction
         mean_prediction = model_prediction(sobol_parameters, self.name, self.scale_factor)
        
         # Initialize lists for Sobol indices List
         ST_list = []
         S1_list = []
         S2_list = []
        
         # Loop over columns for each output corresponding to a single observable for all parameters set
         for col_idx in range(mean_prediction.shape[1]):
             Y = mean_prediction[:, col_idx]
             sobol_indices = sobol.analyze(self.sp, Y)
            
             ST, S1, S2 = sobol_indices.to_df()
            
             ST_list.append(np.array(ST))
             S1_list.append(np.array(S1))
             S2_list.append(np.array(S2))
             if self.print_on:
              print("Total Sobol indices:", ST)
              print("Frist-order Sobol indices:", S1)
              print("Second-order Sobol indices:", S2)

         # Save Sobol indices to file
         sobol_inds=[ST_list, S1_list, S2_list]
         with open(sobol_file, "wb") as f:
             pk.dump(sobol_inds, f)
         print(f"\033[92m {'*'*50}\n  Sobol Indices Generation Done 😊!!, Save in File: sobol_{self.name}.dump\"  \n{'*'*50}\033[0m")  
         return ST_list, S1_list, S2_list
         
      # If Sobol indices file exists, load and return indices           
      else: 
        print(f"\033[92m {'*'*80}\n  File: Found  😊!!, Returning Sobol Indices From \"sobol_{self.name}.dump\"   \n{'*'*80}\033[0m")  
        sobol_file = pk.load(open(sobol_file, "rb"))
        ST_list, S1_list, S2_list=sobol_file
        return ST_list, S1_list, S2_list
         

 
