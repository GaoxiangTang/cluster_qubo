from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.algorithms.minimum_eigensolvers import QAOA
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
import numpy as np
from qiskit_aer.primitives import Sampler
from qiskit.primitives import BackendEstimator, BackendSampler
import time
import random
import math
import heapq


def solve_classical(qubo_matrix):
    n = qubo_matrix.shape[0]  # 矩阵大小，即问题的变量数
    best_solution = None
    best_value = float('inf')
    
    # 枚举所有可能的解。对于n个变量的问题，有2^n种可能的解向量
    for i in range(2**n):
        # 生成当前解向量
        solution = [(i >> bit) & 1 for bit in range(n)]
        solution_array = np.array(solution)
        
        # 计算当前解的目标函数值
        value = solution_array @ qubo_matrix @solution_array
        
        # 更新最优解
        if value < best_value:
            best_solution = solution_array
            best_value = value
    
    return best_solution, best_value


import numpy as np
from tabu import TabuSampler

def solve_tabu(Q_matrix: np.array, initial_guess: np.array = None):
    # 将Q_matrix转换为dict格式，符合D-Wave输入要求
    Q_dict = {(i, j): Q_matrix[i, j] for i in range(len(Q_matrix)) for j in range(i, len(Q_matrix))}
    
    # 初始化Tabu采样器
    sampler = TabuSampler()
    
    # 进行采样，考虑初始猜测
    if initial_guess is not None:
        initial_states = [initial_guess.tolist()]
        response = sampler.sample_qubo(Q_dict, initial_states=initial_states)
    else:
        response = sampler.sample_qubo(Q_dict)
    
    # 获取最低能量状态对应的解
    solution = response.first.sample
    
    # 将解转换为二进制向量
    binary_vector = np.array([solution[i] for i in range(len(Q_matrix))])
    
    return binary_vector, evaluate_qubo(Q_matrix, binary_vector)


def solve_qubo_with_qaoa(matrix, reps=1, shots=None, initial_guess=None, noise_model=None, name=None, detail_output=False):
    """
    Solves a QUBO problem using Qiskit's QAOA implementation on the given backend.  
    
    
    Parameters:
    matrix (numpy.ndarray): An upper-triangle matrix representing the QUBO problem.
    reps (int): number of layers of the QAOA ansatz.
    backend: Qiskit backend on which the QAOA algorithm runs on.
    
    Returns:
    np.ndarray: Solution vector of the QUBO problem.
    """
    # Define the QUBO problem
    qp = QuadraticProgram()
    for i in range(matrix.shape[0]):
        qp.binary_var(f'x{i}')
    qp.minimize(quadratic=matrix, )

    # Convert to QUBO format
    qp2qubo = QuadraticProgramToQubo()
    qubo = qp2qubo.convert(qp)

    # Set up the quantum instance (sampler) using a simulator
    sampler = Sampler(backend_options={"noise_model": noise_model, "shots": shots})

    # Configure QAOA with the sampler
    # name = initialize_callback(name)
    optimizer = COBYLA()
    qaoa = QAOA(sampler=sampler, initial_point=initial_guess, optimizer=optimizer, reps=reps)
    

    # Solve the QUBO problem using QAOA
    qaoa_optimizer = MinimumEigenOptimizer(qaoa)
    result = qaoa_optimizer.solve(qubo)

    # print(qaoa.optimal_params, qaoa.ansatz)
    # print(result.min_eigen_solver_result.cost_function_evals)

    if not detail_output:
        return result.x
    return result, qaoa.ansatz


def local_search(Q, initial_guess):
    """
    Perform a local search to find a local minimum of a QUBO problem.

    Parameters:
    - Q: The QUBO matrix.
    - initial_guess: An initial solution guess as a binary list.
    Returns:
    - best_solution: The binary vector representing the best solution found.
    - best_value: The value of the QUBO objective function at the best solution.
    """

    # Initialize the best solution and its value
    best_solution = np.array(initial_guess)
    best_value = best_solution.T @ Q @ best_solution

    n = len(initial_guess)  # Number of variables
    improvement = True

    while improvement:
        improvement = False
        for i in range(n):
            delta = np.array([Q[i, i] if j == i else best_solution[j]* (Q[i, j] + Q[j, i]) for j in range(n)]).sum() * (1 if best_solution[i]==0 else -1)
            if delta < 0:
                best_value += delta
                best_solution[i]^=1
                improvement=True

    return best_solution

def evaluate_qubo(matrix, solution):
    """
    Calculate the value of the QUBO problem for a given solution.

    :param matrix: 2D numpy array representing the QUBO matrix.
    :param solution: 1D numpy array representing the solution vector.
    :return: The value of the QUBO problem for the given solution.
    """
    # Since the QUBO is a quadratic binary model, we can calculate the value directly
    # as the solution vector transposed, times the QUBO matrix, times the solution vector.
    # This is equivalent to the dot product of the solution with the matrix-vector product.
   
    return solution @ matrix @ solution


def impact(matrix, x):
    """
    Calculate the impact of each variable in the solution.

    :param matrix: 2D numpy array representing the QUBO matrix.
    :param x: 1D numpy array representing the current solution.
    :return: A numpy array of impacts for each variable.
    """
    
    # Initialize the impact array
    impacts = np.zeros(len(x))
    
    # Calculate the impact of flipping each variable
    for i in range(len(x)):
        # Calculate the change in energy if variable i is flipped
        delta_energy = 0
        for j in range(len(x)):
            if i != j:
                delta_energy += (matrix[i, j] + matrix[j, i]) * x[j]
            else:
                delta_energy += matrix[i, i]
                
        # Impact is the change in energy if the variable is negated
        # Since we're assuming the current solution is a local minimum, the value will not decrease
        impacts[i] = delta_energy if x[i] == 0 else -delta_energy
    
    return impacts

def sub_qubo_matrix(matrix, x, indices):
    """
    Generate a sub-QUBO matrix for a subset of indices, clamping the variables outside of the subset.

    :param matrix: 2D numpy array representing the QUBO matrix.
    :param x: 1D numpy array representing the current solution vector.
    :param indices: list or array of indices that need to be optimized.
    :return: A 2D numpy array representing the sub-QUBO matrix.
    """
    n = len(x)
    sub_matrix = np.zeros((len(indices), len(indices)))

    # Calculate the contribution to the linear term from the variables that are clamped
    d = np.zeros(n)
    offset = 0
    for i in range(n):
        for j in range(n):
            if i not in indices and j not in indices:
                offset += x[i]*x[j]*matrix[i, j]
            if j not in indices:
                d[i] += (matrix[i, j] + matrix[j, i]) * x[j]

    # Create the sub-QUBO matrix
    for idx_i, i in enumerate(indices):
        for idx_j, j in enumerate(indices):
            if i == j:  # Diagonal elements
                sub_matrix[idx_i, idx_i] = matrix[i, i] + d[i]
            else:  # Off-diagonal elements
                sub_matrix[idx_i, idx_j] = matrix[i, j]
    
    return sub_matrix, offset


def group_indices(matrix, best_solution, method, max_qubit_size, min_ratio=None, pool=[], N_S=None, multi_view=None):
    if method == "impact":
        return group_by_impact(matrix, best_solution, max_qubit_size)
    if method == "cluster":
        return group_by_clustering(matrix, best_solution, max_qubit_size, min_ratio, multi_view)
    if method == "random":
        return group_random(len(matrix), max_qubit_size)
    if method == "pool":
        return group_pool(len(matrix), max_qubit_size, pool, N_S)

def group_random(n, max_qubit_size):
    numbers = list(range(n))
    random.shuffle(numbers)
    num_groups = math.ceil(n / max_qubit_size)
    groups = []
    for i in range(num_groups):
        group = numbers[i*max_qubit_size:(i+1)*max_qubit_size]
        groups.append(group)
    return groups

def group_pool(n, max_qubit_size, pool, N_S):
    """
    Group indices by the degree of certainty based on the algorithm described in
    Atobe, Yuta, Masashi Tawada, and Nozomu Togawa. "Hybrid annealing method based on 
    subQUBO model extraction with multiple solution instances." IEEE Transactions on 
    Computers 71.10 (2021): 2606-2619.
    
    Parameters:
    - n: The length of the solution vectors.
    - max_qubit_size: Maximum size of each group.
    - pool: List of solution vectors.
    - N_S: Maximum number of solutions to consider from the pool.
    
    Returns:
    - List of groups (lists of indices) sorted by degree of certainty.
    """
    
    # Randomly select min(len(pool), N_S) elements from the pool
    sample_size = min(len(pool), N_S)
    sample_set = random.sample(pool, sample_size)
    
    # Initialize the count array
    count_array = np.zeros(n)
    
    # Calculate c_i for each index i
    for neg_funcval, solution in sample_set:
        count_array += solution
    
    # Calculate the degree of certainty d_i for each index i
    d_i = np.abs((sample_size / 2) - count_array)
    
    # Get the sorted indices based on the degree of certainty
    sorted_indices = np.argsort(d_i)
    
    # Group indices into groups of max_qubit_size
    groups = [
        sorted_indices[i:min(i + max_qubit_size, n)].tolist()
        for i in range(0, n, max_qubit_size)
    ]
    
    return groups


def group_by_impact(matrix, best_solution, max_qubit_size):
    impacts = impact(matrix, best_solution)
    sorted_indices = np.argsort(impacts)

    groups = []
    for start in range(0, len(sorted_indices), max_qubit_size):
        end = min(start + max_qubit_size, len(sorted_indices))
        groups.append(sorted_indices[start:end].tolist())
    return groups

from src.split_cluster import split_spectral_cluster

def group_by_clustering(matrix, x, max_qubit_size, min_ratio=None, multi_view=None):
    q_tilde = np.outer((-1)**x, (-1)**x) * matrix
    if min_ratio is None:
        min_ratio = 0.8
    labels = split_spectral_cluster(q_tilde, max_qubit_size, max_qubit_size*min_ratio, multi_view=multi_view)
    unique_labels = np.unique(labels)
    groups = [list(np.where(labels == label)[0]) for label in unique_labels]
    return groups

def select_subsets(groups, fraction, n, shuffle=True):
    if shuffle:
        np.random.shuffle(groups)
    selected_groups = []
    total_elements = 0

    for group in groups:
        selected_groups.append(group)
        total_elements += len(group)
        if total_elements > fraction * n:
            break
    
    return selected_groups

from copy import copy

def hybrid_qubo_solve(matrix, num_repeats=100, fraction=0.15, max_qubit_size=20,
                      group_method="impact", classical_method='local', N_S=5, N_I=20, multi_view=None,
                      subproblem_classical=False):
    """
    Hybrid solver for QUBO problems using both classical Tabu Search and quantum QAOA, based on Algorithm 1 from the article.
    
    :param matrix: 2D numpy array representing the QUBO matrix.
    :param num_repeats: Number of iterations for the hybrid algorithm.
    :param max_qubit_size: The maximum size of the sub-QUBO problems due to hardware constraints.
    :return: Tuple of (1D numpy array representing the best solution found, metadata dictionary).
    """
    n = len(matrix)
    quantum_calls = 0
    total_iters = 0
    quantum_time = 0
    start_time = time.time()

    classical_values = []  # List to store function values after each classical optimization step

    # Initialize with tabu search
    if classical_method == 'local':
        best_solution = local_search(matrix, np.random.randint(0, 2, n))
        best_value = evaluate_qubo(matrix, best_solution)
    elif classical_method == 'tabu':
        best_solution, best_value = solve_tabu(matrix)
        print('initial', best_value)
    current_solution = copy(best_solution)
    no_improvement_streak = 0  # Counter for iterations without improvements
    pool = None
    if group_method == 'pool':
        pool = []
        heapq.heappush(pool, (-best_value, tuple(best_solution)))
    
    classical_values.append(best_value)  # Record the initial function value

    # Main loop
    repeats = 0
    while no_improvement_streak <= 3 and repeats < num_repeats:
        
        repeats += 1
        groups = group_indices(matrix, best_solution, group_method, max_qubit_size, pool=pool, N_S=N_S, multi_view=multi_view)
        # 'pool' method selects the least determined solution vectors.
        selected_groups = select_subsets(groups, fraction, n, group_method != 'pool')
        
        for indices in selected_groups:
            sub_matrix, offset = sub_qubo_matrix(matrix, current_solution, indices)
            
            # Record start time of quantum computation
            quantum_start = time.time()
            quantum_calls += 1
            if subproblem_classical:
                qaoa_solution, _ = solve_tabu(sub_matrix)
            else:
            
                # Solve the sub-QUBO using QAOA
                res, ansatz = solve_qubo_with_qaoa(sub_matrix, detail_output=True) 
                total_iters += res.min_eigen_solver_result.cost_function_evals
                
                # Record the time spent on quantum computation
                quantum_time += time.time() - quantum_start
                
                qaoa_solution = res.x

            # Update the solution with the result from QAOA
            # if evaluate_qubo(sub_matrix, qaoa_solution) < evaluate_qubo(sub_matrix, current_solution[indices]):
            #     print("new minimum found in QAOA", evaluate_qubo(sub_matrix, qaoa_solution), " lesser than ", evaluate_qubo(sub_matrix, current_solution[indices]))
            current_solution[indices] = qaoa_solution

        if classical_method == 'local':
            current_solution = local_search(matrix, current_solution)
            current_value = evaluate_qubo(matrix, current_solution)
        elif classical_method == 'tabu':
            current_solution, current_value = solve_tabu(matrix, current_solution) 

        classical_values.append(current_value)  # Record the function value after classical optimization

        if group_method == 'pool':
            heapq.heappush(pool, (-current_value, tuple(current_solution)))
            if len(pool) > N_I:
                heapq.heappop(pool)

        if current_value < best_value:
            best_solution = np.copy(current_solution)
            best_value = current_value
            no_improvement_streak = 0
        else:
            no_improvement_streak += 1

    total_runtime = time.time() - start_time
    classical_runtime = total_runtime - quantum_time
    result = {
        'x': best_solution,
        'val': evaluate_qubo(matrix, best_solution),
        'quantum_calls': quantum_calls,
        'total_iters': total_iters,
        'classical_runtime': classical_runtime,
        'classical_values': classical_values  # Include the recorded function values in the result
    }

    return result

 


