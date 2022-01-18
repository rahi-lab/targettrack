import networkx as nx
import networkx.algorithms.components as nxc
import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from scipy.sparse import csr_matrix
from scipy.sparse import spdiags
from scipy.sparse.linalg import eigs
from scipy.sparse import lil_matrix
from matplotlib import pyplot as plt
import logging


LOGGER = logging.getLogger("GBC")

def nonoise_score(cluster, truth, noise_ix):
    """
    Calculates adjusted rand score on all non-noise elements.
    """
    keep = np.logical_not(noise_ix)
    return adjusted_rand_score(cluster[keep], truth[keep])


def graph_impute(data, nneighbors):
    """
    Imputes the graph with k-nearest-neighbors method, returns
    adjacency matrix
    """
    return kneighbors_graph(data, nneighbors, mode='connectivity')


def disconnect_same_time(adj, time):
    """
    Remove all edges that link two pieces that are from the same timeframe
    """
    out = adj.todense()
    outerprod = np.outer(time,time)
    uniq_t = np.unique(time)
    for t in np.unique(time):
        out[outerprod==time]=0
    # out.eliminate_zeros()
    return csr_matrix(out)


def compute_spectral_embedding(adj, k):
    """
    Returns smallest k eigenvalues and eigenvectors of graph Laplacian
    """
    n = adj.shape[0]
    d = spdiags(adj.sum(axis=0), 0, n, n)
    L = d-adj
    return eigs(L, k, which='SM')

    
def spectral_cluster(evecs, k):
    """
    Clusters the nodes with k-means clustering applied on the first k eigenvectors
    """
    kmeans = KMeans(n_clusters=k).fit(evecs[:,:k])
    return kmeans.labels_


def number_subpartitions(conncomp, time, tolerance=.1):
    """
    Determines which connected component to subcluster, and into how many
    subclusters, as follows: Get the times that correspond to a given connected
    component. For every distinct time, the number of its occurrence is counted. 
    The number of subclusters is the average count minus the tolerance.
    
    Parameters
    ----------
    conncomp:
        List of sets of integers, as output by list(nx.connected_components)
    time:
        Time steps corresponding to every segment
    tolerance:
        Subtracted from the average occurence
        
    Returns
    -------
    n_segs:
        Number of segments for every connected component
    """
    n_comp = len(conncomp)
    n_segs = np.ones(n_comp)
    for i in range(n_comp):
        seg_times = time[list(conncomp[i])]
        _, counts = np.unique(seg_times, return_counts=True)
        n_segs[i] = np.ceil(counts.mean() - tolerance)
            
    return n_segs.astype(int)


def graph_based_cluster(data, time, nneighbors=20, verbose=False):
    """
    Clusters data using an imputed graph with knn, with edges between segments
    of the same timeframe removed. First splits into connected components,
    then determines for every connected component how many subclusters we want to
    find. Subclustering is done with spectral clustering.
    
    Parameters
    ----------
    data:
        Features to use for graph imputation
    time:
        For every segment specifies its time
    nneighbors:
        Number of neighbors used for knn-graph imputation

    Returns
    -------
    clusters:
        Cluster assignment for every segment
    """
    
    # Inner function that handles subclustering
    def do_cluster(adj, n):
        if n==1:
            return np.zeros(adj.shape[0])
        _, evecs = compute_spectral_embedding(adj, n)
        return spectral_cluster(np.real(evecs), n)

    
    # Impute graph, get connected components
    adj = graph_impute(data, nneighbors)
    disconnected = disconnect_same_time(adj, time)
    nx_graph = nx.from_scipy_sparse_matrix(disconnected)
    
    conncomp = list(nx.connected_component_subgraphs(nx_graph))
    conncomp_nodes = [list(x.nodes) for x in conncomp]
    
    # Subpartition
    n_subpart = number_subpartitions(conncomp_nodes, time)
    if verbose:
        LOGGER.info("Number of nodes:", [len(x) for x in conncomp_nodes])
        LOGGER.info("Number subpartitions:", n_subpart)
    
    # Create subpartitions using spectral clustering
    subparts = [do_cluster(nx.adjacency_matrix(x), n) for x, n in zip(conncomp, n_subpart)]
    clusters = np.zeros(len(time))
    for i, subpart in enumerate(subparts):
        maxcluster = clusters.max()
        clusters[conncomp_nodes[i]] = subpart + maxcluster + 1
    
    return clusters


