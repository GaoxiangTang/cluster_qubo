# %%
from src.hybrid_qubo import *
from src.split_cluster import *

# %%
import pickle
import numpy as np
from tqdm import tqdm

def correct_data(data_list):
    for entry in tqdm(data_list): 
        qubo_matrix = entry['matrix']
        
        results_cluster_coreg = {}
        for qubit_size in range(10, 25):
            print(qubit_size)
            res_cluster_coreg = hybrid_qubo_solve(
                qubo_matrix, 
                max_qubit_size=qubit_size, 
                group_method='cluster', 
                multi_view='co-regularization',
                simulated=True,
            )
            results_cluster_coreg[qubit_size] = res_cluster_coreg

        entry['results_cluster_coreg'] = results_cluster_coreg
    
    return data_list

def main():
    # Load the existing data
    with open('data/100er_d_data.pkl', 'rb') as file:
        data_list = pickle.load(file)
    
    # Correct the data
    corrected_data_list = correct_data(data_list)
    
    # Save the updated data
    with open('data/100er_d_data.pkl', 'wb') as file:
        pickle.dump(corrected_data_list, file)

if __name__ == "__main__":
    main()