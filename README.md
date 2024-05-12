# HepBayes
This module serves as a comprehensive toolkit for researchers and practitioners engaged in data analysis,   modeling, and simulation in various scientific domains.


A. To run the code successfully, ensure you have the following prerequisites installed:

Python (version 3.x recommended)
Required Python libraries:
1.  os
2.  sys
3.  numpy  
4.  pandas  
5.  pickle  
6.  seaborn  
7.  matplotlib 
8.  sklearn
9.  corner
10. emcee
11. scipy
12. termcolor
13. io
14.SALib
#######################################################################################################
#                             Steps to Run the Code                                                   #
#######################################################################################################
B. 	                Generate  and Parameter Set Model Output
#-----------------------------------------------------------------------------------------------------#
	I.	Generate Training Parameter Set: Run LHS_DesignGenerator to generate the training parameter set.
	II.	Generate Training Model Set: Generate model output for each observables in the training parameter set (training_model_set).
	III.	Generate Testing Parameter Set: Again, run LHS_DesignGenerator to generate a smaller number of parameter sets (testing_parameter_set).
	IV.	Generate Testing Model Set: Repeat step C for the testing parameter set to create the testing model set.
#######################################################################################################
C.				Prepare Input Files:
#######################################################################################################
1.					For Model:
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
i)	Training parameter set (Dimensions:Number of training points × Number of Peremeters )
ii)	Training model set (Dimensions: Number of training points X number of Observable)
       (i,e, Arrange the model outputs into a matrix where each column represents the same observable for all training points.)
iii)	Testing parameter set (Dimensions: Same as training parameter set)
iv)	Testing model set (Dimensions: Same as training model set)

#######################################################################################################
2.					For Experimental Data::
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
1.	All experimental results (should have X-value, Y-value, Y-error)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Note: You can have multiple files for training parameter sets and experimental results. For example, 
you can have separate files for pT spectra for pion, kaon, and proton.
File names can be anything, but "training" and "testing" should be included in the names.

# ALL DONE!!!, PUT EVERYTHING (ALL FILES) IN INPUT FOLDER"
JUST run example/Bayesian_inference.py.
Remember to set the values of other variables in Bayesian_inference.py according to your requirements.
"HOPEFULLY YOU WILL SEE SOME BEAUTIFUL PLOTS"






