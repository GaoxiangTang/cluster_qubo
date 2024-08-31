# %%
from src.hybrid_qubo import *
from src.split_cluster import *

# %%
import numpy as np
import networkx as nx

def gen_graph(N, D, g, weight_func=lambda: np.random.randint(1, 10), max_attempts=1000):
    """
    Generate a D-regular graph with N nodes and a girth greater than g.
    If the first attempt doesn't meet the girth criteria, keep trying up to max_attempts times.
    Edges are weighted using the specified weight function.

    Parameters:
    D (int): Degree of each node.
    N (int): Number of nodes in the graph.
    g (int): Minimum girth of the graph.
    weight_func (callable, optional): Function to generate edge weights. Defaults to a random integer [1, 10).
    max_attempts (int): Maximum number of attempts to generate a graph meeting the criteria.

    Returns:
    graph: A NetworkX graph object meeting the specified criteria, or None if no such graph can be generated.
    """
    attempt = 0
    while attempt < max_attempts:
        try:
            graph = nx.random_regular_graph(D, N)
            if nx.cycle_basis(graph):
                shortest_cycle = min(len(cycle) for cycle in nx.cycle_basis(graph))
                if shortest_cycle > g:
                    # Assign weights to the edges
                    for (u, v) in graph.edges():
                        graph[u][v]['weight'] = weight_func()
                    return graph
        except nx.NetworkXError as e:
            print(f"Error generating a {D}-regular graph with {N} nodes on attempt {attempt+1}: {e}")
        attempt += 1

    print(f"Failed to generate a graph meeting the criteria after {max_attempts} attempts.")
    return None

import networkx as nx
import numpy as np

def generate_er_graph(n, p, weight_function=None):
    """
    Generate an Erdős–Rényi graph G(n, p) with weights.

    Parameters:
    - n (int): Number of nodes in the graph.
    - p (float): Probability of an edge between any two nodes.
    - weight_function (callable, optional): Function to generate edge weights.
      If None, uses random integers between -10 and 10.

    Returns:
    - G (networkx.Graph): The generated graph with weighted edges.
    """
    # 默认的权重生成函数
    if weight_function is None:
        weight_function = lambda: np.random.randint(1, 10)
    
    # 创建一个空的无向图
    G = nx.Graph()
    
    # 添加节点
    G.add_nodes_from(range(n))
    
    # 添加边
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.rand() <= p:  # 使用均匀分布来决定是否添加边
                # 如果边存在，使用权重函数来赋予权重
                G.add_edge(i, j, weight=weight_function())
    
    return G

# 例子：生成一个5个节点的图，边存在的概率为0.5
graph = generate_er_graph(5, 0.5)
print(nx.info(graph))



# %%
import numpy as np
import pickle

def save_data(data_list, filename='qubo_data.pkl'):
    with open(filename, 'wb') as f:
        # Serializing the data list using pickle
        pickle.dump(data_list, f)

def load_data(filename='qubo_data.pkl'):
    with open(filename, 'rb') as f:
        # Deserializing the data from the file
        data_list = pickle.load(f)
    return data_list

# import numpy as np
# import pickle
# from qiskit_optimization.applications import Maxcut
# from tqdm import tqdm

# data_dict = {}
# graph_sizes = [80, 120, 140, 160, 180]
# delta = 0

# for size in graph_sizes:
#     data_list = []
#     for _ in tqdm(range(80), desc=f"Processing size {size}"):  # tqdm will display the progress
#         G = gen_graph(size, 3, 3)
#         qubo = Maxcut(G).to_quadratic_program()

#         variables = qubo.variables
#         qubo_matrix = qubo.objective.quadratic.to_array()
#         linear_terms = qubo.objective.linear.to_array()

#         np.fill_diagonal(qubo_matrix, qubo_matrix.diagonal() + linear_terms)
#         qubo_matrix *= -1

#         results_impact = {}
#         results_cluster_concat = {}
#         results_cluster_coreg = {}
#         results_random = {}
#         results_pool = {}

#         for qubit_size in range(10, 25):
#             res_impact = hybrid_qubo_solve(qubo_matrix, max_qubit_size=qubit_size, group_method='impact', simulated=True)
#             res_cluster_concat = hybrid_qubo_solve(qubo_matrix, max_qubit_size=qubit_size, group_method='cluster', simulated=True)
#             # res_cluster_coreg = hybrid_qubo_solve(qubo_matrix, max_qubit_size=qubit_size, group_method='cluster', multi_view='co-regularization')
#             # res_random = hybrid_qubo_solve(qubo_matrix, max_qubit_size=qubit_size, group_method='random', simulated=True, classical_method='tabu')
#             # res_pool = hybrid_qubo_solve(qubo_matrix, max_qubit_size=qubit_size, group_method='pool', simulated=True, classical_method='tabu')

#             results_impact[qubit_size] = res_impact
#             results_cluster_concat[qubit_size] = res_cluster_concat
#             # results_cluster_coreg[qubit_size] = res_cluster_coreg
#             # results_pool[qubit_size] = res_pool
#             # print('pool', res_pool['val'])
#             # results_random[qubit_size] = res_random
#             # print('random', res_random['val'])
#             # delta += res_pool['val'] - res_random['val']
#             # print('delta', delta)

#         data_list.append({
#             'matrix': qubo_matrix,
#             'results_impact': results_impact,
#             'results_cluster': results_cluster_concat,
#             # 'results_cluster_coreg': results_cluster_coreg,
#             # 'results_random': results_random,
#             # 'results_pool': results_pool
#         })
    
#     data_dict[size] = data_list

#     # Save the updated data list for the current size
#     # with open(f'{size}Node3Regular_d_data_scale.pkl', 'wb') as file:
#     #     pickle.dump(data_dict, file)

# # Optionally, save the complete data dictionary for all sizes
# with open('AllSizesNode3Regular_d_data_scale_impact.pkl', 'wb') as file:
#     pickle.dump(data_dict, file)


import numpy as np
import pickle
from qiskit_optimization.applications import Maxcut
from tqdm import tqdm

# Load existing data if available
try:
    with open('AllSizesNode3Regular_d_data_scale.pkl', 'rb') as file:
        data_dict = pickle.load(file)
except FileNotFoundError:
    # data_dict = {}
    raise()

# Updated graph sizes
additional_graph_sizes = [100]
additional_qubit_size_range = range(5, 25)

# Existing graph sizes
existing_graph_sizes = [80, 120, 140, 160, 180]
existing_qubit_size_range = range(5, 10)

# Process additional graph sizes
for size in additional_graph_sizes:
    if size not in data_dict:
        data_dict[size] = []

    for _ in tqdm(range(80), desc=f"Processing size {size}"):
        G = gen_graph(size, 3, 3)
        qubo = Maxcut(G).to_quadratic_program()

        variables = qubo.variables
        qubo_matrix = qubo.objective.quadratic.to_array()
        linear_terms = qubo.objective.linear.to_array()

        np.fill_diagonal(qubo_matrix, qubo_matrix.diagonal() + linear_terms)
        qubo_matrix *= -1

        results_impact = {}
        results_cluster_concat = {}

        for qubit_size in additional_qubit_size_range:
            res_impact = hybrid_qubo_solve(qubo_matrix, max_qubit_size=qubit_size, group_method='impact', simulated=True)
            res_cluster_concat = hybrid_qubo_solve(qubo_matrix, max_qubit_size=qubit_size, group_method='cluster', simulated=True)

            results_impact[qubit_size] = res_impact
            results_cluster_concat[qubit_size] = res_cluster_concat

        data_dict[size].append({
            'matrix': qubo_matrix,
            'results_impact': results_impact,
            'results_cluster': results_cluster_concat,
        })

# Process additional qubit sizes for existing graph sizes
for size in existing_graph_sizes:
    for data_entry in tqdm(data_dict[size], desc=f"Updating size {size}"):
        qubo_matrix = data_entry['matrix']

        results_impact = data_entry.get('results_impact', {})
        results_cluster_concat = data_entry.get('results_cluster', {})

        for qubit_size in existing_qubit_size_range:
            if qubit_size not in results_impact:
                res_impact = hybrid_qubo_solve(qubo_matrix, max_qubit_size=qubit_size, group_method='impact', simulated=True)
                results_impact[qubit_size] = res_impact

            if qubit_size not in results_cluster_concat:
                res_cluster_concat = hybrid_qubo_solve(qubo_matrix, max_qubit_size=qubit_size, group_method='cluster', simulated=True)
                results_cluster_concat[qubit_size] = res_cluster_concat

        data_entry['results_impact'] = results_impact
        data_entry['results_cluster'] = results_cluster_concat

# Save the complete data dictionary
with open('AllSizesNode3Regular_d_data_scale.pkl', 'wb') as file:
    pickle.dump(data_dict, file)
