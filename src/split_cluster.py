import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from scipy.linalg import eigh
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import linkage, fcluster
from k_means_constrained import KMeansConstrained
import numpy as np

def agglomerative_cluster(f_matrix, max_size):
    """
    Performs agglomerative hierarchical clustering on a feature matrix, ensuring each cluster's maximum size does not exceed max_size.
    
    Parameters:
    - f_matrix: Feature matrix where each row is a sample and each column is a feature.
    - max_size: The maximum allowed size for each cluster.
    
    Returns:
    - cluster_assignments: An array where each element is the cluster number assigned to the corresponding sample.
    """
    # Perform hierarchical clustering using the ward method.
    Z = linkage(f_matrix, 'ward')
    
    # Initialize cluster assignments, initially each sample is its own cluster.
    cluster_assignments = np.arange(len(f_matrix))
    
    # Check each merge operation.
    for i, merge in enumerate(Z[:, :2].astype(int)):
        cluster_size = np.sum(np.isin(cluster_assignments, merge))
        
        # If the merged cluster's size is less than or equal to the maximum allowed size, perform the merge.
        if cluster_size <= max_size:
            np.put(cluster_assignments, np.where(np.isin(cluster_assignments, merge)), len(f_matrix) + i)
    
    # Ensure cluster numbers are continuous.
    unique_clusters = np.unique(cluster_assignments)
    continuous_cluster_assignments = np.zeros_like(cluster_assignments)
    for i, unique_cluster in enumerate(unique_clusters):
        continuous_cluster_assignments[cluster_assignments == unique_cluster] = i
    
    return continuous_cluster_assignments

def divisive_cluster(X, max_size):
    """
    Performs divisive hierarchical clustering on a feature matrix, ensuring each cluster's maximum size does not exceed max_size.
    
    Parameters:
    - X: The data matrix where each row is a sample.
    - max_size: The maximum allowed size for each cluster.
    
    Returns:
    - cluster_labels: An array of cluster labels for each sample.
    """
    # Initialize cluster labels.
    cluster_labels = np.zeros(X.shape[0], dtype=int)
    next_cluster_label = 1  # The next available cluster number.
    
    # Initialize a list of clusters, each element is a set of sample indices.
    clusters_to_split = [np.arange(X.shape[0])]
    
    while clusters_to_split:
        current_cluster_indices = clusters_to_split.pop(0)
        if len(current_cluster_indices) <= max_size:
            # If the current cluster's size is less than or equal to max_size, assign it a number.
            cluster_labels[current_cluster_indices] = next_cluster_label
            next_cluster_label += 1
        else:
            # Split the current cluster.
            kmeans = KMeans(n_clusters=2, random_state=42).fit(X[current_cluster_indices])
            labels = kmeans.labels_
            
            # Add new clusters for splitting.
            for label in np.unique(labels):
                indices = current_cluster_indices[labels == label]
                clusters_to_split.append(indices)
    
    return cluster_labels

def constrained_kmeans(X, max_size, min_size, n_clusters):
    """
    Clusters feature vectors using a constrained k-means algorithm.
    
    Parameters:
    - X: The data matrix where each row is a sample.
    - max_size: The maximum allowed size for each cluster.
    
    Returns:
    - An array of cluster labels for each sample.
    """
    n = len(X)
    kmeans = KMeansConstrained(
        n_clusters=n_clusters,
        size_max=max_size,
        size_min=min_size,
    )
    return kmeans.fit_predict(X)

from math import ceil, floor
from sklearn.metrics import normalized_mutual_info_score

def compute_normalized_laplacian(affinity_matrix):
    """Compute the normalized Laplacian matrix from the affinity matrix."""
    degree_matrix = np.diag(np.sum(affinity_matrix, axis=1))
    # Handle zero degrees to avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        d_inv_sqrt = np.diag(1.0 / np.sqrt(np.sum(affinity_matrix, axis=1)))
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
        d_inv_sqrt[np.isnan(d_inv_sqrt)] = 0
    
    laplacian_matrix = degree_matrix - affinity_matrix
    normalized_laplacian = np.matmul(np.matmul(d_inv_sqrt, laplacian_matrix), d_inv_sqrt)
    return normalized_laplacian

def get_k_smallest_eigenvectors(laplacian, k):
    """Get the k smallest eigenvectors of the Laplacian matrix."""
    eigvals, eigvecs = np.linalg.eigh(laplacian)
    sorted_indices = np.argsort(eigvals)
    k_smallest_indices = sorted_indices[:k]
    return eigvecs[:, k_smallest_indices]

def co_regularization(views, number_of_clusters, number_of_iterations=1000, lambda_value=0.025, random_state=None, calculate_nmi=False, true_labels=None):
    """
    Co-Regularized Multi-view Spectral Clustering algorithm modified to return the first view's feature matrix and NMI over time if specified.
    """
    L = {}
    U = {}
    val = 0
    for i, view in enumerate(views):
        sim_matrix = view
        L[i] = compute_normalized_laplacian(sim_matrix)
        U[i] = get_k_smallest_eigenvectors(L[i], number_of_clusters)
        val += np.trace(U[i].T @ L[i] @ U[i])
        for j in range(i):
            val -= lambda_value * np.trace(U[i].T @ U[j] @ U[j].T @ U[i])

    vals = [val]
    nmi_scores = []

    if calculate_nmi and true_labels is not None:
        kmeans = KMeans(n_clusters=number_of_clusters, random_state=random_state).fit(U[0])
        predicted_labels = kmeans.labels_
        nmi_score = normalized_mutual_info_score(true_labels, predicted_labels)
        nmi_scores.append(nmi_score)

    for _ in range(number_of_iterations):
        delta = 0
        for j, view in enumerate(views):
            KU = sum((eigenvectors @ eigenvectors.T) for k, eigenvectors in U.items() if k != j)
            lap = L[j] - lambda_value * KU
            last = np.trace(U[j].T @ lap @ U[j])
            U[j] = get_k_smallest_eigenvectors(lap, number_of_clusters)
            delta += np.trace(U[j].T @ lap @ U[j]) - last
            L[j] =lap
        vals.append(vals[-1] + delta)
        
        if calculate_nmi and true_labels is not None:
            kmeans = KMeans(n_clusters=number_of_clusters, random_state=random_state).fit(U[0])
            predicted_labels = kmeans.labels_
            nmi_score = normalized_mutual_info_score(true_labels, predicted_labels)
            nmi_scores.append(nmi_score)
        
        if abs(delta) < 5e-6:
            break

    if calculate_nmi and true_labels is not None:
        return U[0], vals, nmi_scores
    else:
        return U[0]

def split_spectral_cluster(Q_tilde, max_size, min_size=0, n_clusters=None, method="constrained_kmeans", multi_view=None):
    """
    Clusters the feature matrix Q_tilde into positive and negative parts, calculates their normalized Laplacians, 
    extracts eigenvectors corresponding to smaller eigenvalues. Points with larger components in an eigenvector 
    have a stronger characteristic. Clusters are formed by combining the two sets of eigenvectors and performing 
    clustering on their row vectors using one of the specified methods. The result is a clustering of indices.
    
    Parameters:
    - Q_tilde: The feature matrix.
    - max_size: The maximum allowed size for each cluster.
    - method: The constrained clustering method to use after the feature matrix is obtained, can be "agglomerative", "divisive", or "constrained_kmeans".
    - multi_view: How to combine multi-view correlation matrix into one, choose between 'concatenation' and 'co-regularization'
    
    Returns:
    - An array of cluster labels for each index.
    """

    if multi_view == None:
        multi_view = 'concatenation'

    n = len(Q_tilde)
    S_pos = np.maximum(Q_tilde, 0)
    S_neg = np.maximum(-Q_tilde, 0)
    
    if n_clusters is None:
        if min_size != 0:
            n_clusters = int((ceil(n / max_size) + floor(n / min_size) )/2)
        else:
            n_clusters = int(n / max_size * 1.5)
    
    def get_eigenvectors(S):
        S = (S + S.T) / 2
        D = np.diag(np.sum(S, axis=1))
        L = D - S
        eigvals, eigvecs = eigh(L)
        # print(sorted(eigvals))
        return eigvecs[:, np.argsort(eigvals)[:int(n_clusters)]]
    
    if multi_view == 'concatenation':
        V_pos = get_eigenvectors(S_pos)
        V_neg = get_eigenvectors(S_neg)
        V_combined = np.concatenate((V_pos, V_neg), axis=1)
    elif multi_view == 'co-regularization':
        V_combined = co_regularization([S_pos, S_neg], number_of_clusters=n_clusters, number_of_iterations=20)
    else:
        raise('choose between \'concatenation\' and \'co-regularization\'')

    
    if method == "agglomerative":
        # Use the Agglomerative clustering on combined eigenvectors.
        return agglomerative_cluster(V_combined, max_size)
    elif method == "divisive":
        # Use the Divisive clustering on combined eigenvectors.
        return divisive_cluster(V_combined, max_size)
    elif method == "constrained_kmeans":
        # Use the Constrained KMeans clustering on combined eigenvectors.
        return constrained_kmeans(V_combined, floor(max_size), ceil(min_size), n_clusters)
    else:
        raise ValueError("Unsupported clustering method. Choose from 'agglomerative', 'divisive', or 'constrained_kmeans'.")


def generate_affinity_matrix_with_negative(n_clusters=4, n_samples_per_cluster=50, n_features=2, noise=0.05):
    # np.random.seed(42)  # For reproducibility
    clusters = []
    for i in range(n_clusters):
        # Generate random cluster centers
        cluster_center = np.random.uniform(-1, 1, n_features)
        # Generate samples around the cluster centers
        samples = np.random.randn(n_samples_per_cluster, n_features) * noise + cluster_center
        clusters.append(samples)
    all_samples = np.vstack(clusters)
    
    # Compute the Euclidean distance matrix
    distances = np.sqrt(((all_samples[:, np.newaxis, :] - all_samples[np.newaxis, :, :]) ** 2).sum(axis=2))
    
    # Convert distances to affinity (similarity), then introduce negative values randomly
    sigma = np.mean(distances)  # Adjust this parameter as needed
    affinity_matrix = np.exp(-distances ** 2 / (2 * sigma ** 2))
    
    # Randomly flip signs for a subset of the affinity matrix elements to introduce negative relationships
    sign_flipper = np.random.choice([-1, 1], size=affinity_matrix.shape, p=[0.25, 0.75])  # 25% chance to flip sign
    q_tilde = affinity_matrix * sign_flipper
    
    return q_tilde
