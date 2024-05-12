import os
import numpy as np
import pandas as pd
from scipy.stats import qmc
import Graph

class LHS_DesignGenerator:
    def __init__(self, parameters_name=None, initial_values=None, final_values=None, name=None):
        self.file_name = os.path.join(os.getcwd(), "cache_files", f"saved_lhs_{name}.txt")
        self.parameters_name = parameters_name
        self.initial_values = initial_values
        self.final_values = final_values
        self.name = name

    def get_parameter_values_by_name(self, parameter_name):
        """
        Get the initial and final values of a parameter by its name.

        Args:
        parameter_name (str): The name of the parameter.

        Returns:
        tuple: A tuple containing the initial and final values of the parameter.
               If the parameter name is not found, returns None.
        """
        if parameter_name in self.parameters_name:
            index = self.parameters_name.index(parameter_name)
            return self.initial_values[index], self.final_values[index]
        else:
            return None

    def parinfo(self):
        """
        Return model parameters information as a list of tuples.
        Each tuple contains parameter name, initial value, and final value.
        """
        return list(zip(self.parameters_name, self.initial_values, self.final_values))

    def generate_lhs(self, no_points, no_dim, seed):
        """
        Generate Latin-hypercube samples using scipy.stats.qmc.LatinHypercube.
        """
        sampler = qmc.LatinHypercube(len(self.parameters_name))
        return sampler.random(n=no_points)

    def create_dataframe(self, no_points):
        """
        Create a Pandas DataFrame with scaled parameter values using Latin-hypercube sampling.
        """
        parameters, initial_values, final_values = zip(*self.parinfo())
        parameter_ranges = np.array(final_values) - np.array(initial_values)
        scaled_lhs = initial_values + parameter_ranges * np.array(self.generate_lhs(no_points, len(self.parameters_name), 751783496))
        df = pd.DataFrame(scaled_lhs, columns=parameters)
        return df

    def plot_2d_projection(self, no_points, parlist=None):
        """
        Plot a 2D projection of the generated samples.
        """
        df = self.create_dataframe(no_points)
        Graph.plot_2d_projection_lhs(df, no_points, parlist)

    def generate_input_files(self, no_points, no_dim, parlist=None, plot=True):
        """
        Generate input files and save them 
        """
        if not os.path.exists(self.file_name):
            df = self.create_dataframe(no_points)
            np.savetxt(self.file_name, df.values, fmt='%f')
            if plot:
                self.plot_2d_projection(no_points, parlist)


def main():
    """
    Main function to generate input files and plot 2D projections.
    """
    generator = LHS_DesignGenerator(
        parameters_name=["T", 'BetaS', 'mpow', 'etabys', 'alpha', 'Rf', 'tf'],
        initial_values=[0.105, 0.72, 0.71, 0.11, 0.05, 6.5, 6.5],
        final_values=[0.144, 0.905, 0.975, 0.38, 0.133, 14.5, 16.5],
        name="r"
    )
    parlist = ['T', 'Rf']
    generator.generate_input_files(200, 7, parlist)
    df=generator.create_dataframe(200)
    Graph.plot_2d_projection_lhs(df,200,parlist)


if __name__ == '__main__':
    main()

