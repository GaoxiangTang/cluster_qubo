# %%
from src.hybrid_qubo import *
from src.split_cluster import *

# %%
import numpy as np
import networkx as nx
import pickle
from qiskit_optimization.applications import Maxcut
from tqdm import tqdm
from src.hybrid_qubo import hybrid_qubo_solve


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

# 定义图的大小和量子比特范围
graph_sizes = [80, 100, 120, 140, 160, 180]
qubit_size_range = range(5, 25)

# 初始化或加载数据字典
try:
    with open('AllSizesNode3Regular_d_data_scale.pkl', 'rb') as file:
        data_dict = pickle.load(file)
except FileNotFoundError:
    data_dict = {}


# 处理每个图大小
for size in graph_sizes:
    if size not in data_dict:
        data_dict[size] = []
    
    # 如果该大小的数据不足80个,则继续添加
    while len(data_dict[size]) < 80:
        G = gen_graph(size, 3, 3)
        qubo = Maxcut(G).to_quadratic_program()
        
        variables = qubo.variables
        qubo_matrix = qubo.objective.quadratic.to_array()
        linear_terms = qubo.objective.linear.to_array()
        
        np.fill_diagonal(qubo_matrix, qubo_matrix.diagonal() + linear_terms)
        qubo_matrix *= -1
        
        results_impact = {}
        results_cluster = {}
        results_pool = {}
        
        for qubit_size in tqdm(qubit_size_range, desc=f"Processing size {size}, graph {len(data_dict[size])+1}"):
            res_impact = hybrid_qubo_solve(qubo_matrix, max_qubit_size=qubit_size, group_method='impact', simulated=False)
            res_cluster = hybrid_qubo_solve(qubo_matrix, max_qubit_size=qubit_size, group_method='cluster', simulated=False)
            res_pool = hybrid_qubo_solve(qubo_matrix, max_qubit_size=qubit_size, group_method='pool', simulated=False)
            
            results_impact[qubit_size] = res_impact
            results_cluster[qubit_size] = res_cluster
            results_pool[qubit_size] = res_pool
        
        data_dict[size].append({
            'matrix': qubo_matrix,
            'results_impact': results_impact,
            'results_cluster': results_cluster,
            'results_pool': results_pool
        })
        
        # 每处理完一个图就保存一次数据
        with open('AllSizesNode3Regular_d_data_scale.pkl', 'wb') as file:
            pickle.dump(data_dict, file)
