import logging
import warnings
import abc
import src.helpers as h
from src.mask_processing.assignment_finders import AssignmentFinderInterface

# for k-means range clustering
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

# for graph clustering
from src.mask_processing.graph_based_clustering import graph_based_cluster


class Clustering(AssignmentFinderInterface):
    """
    Class to find assignments by clustering.
    Has an instance of the ClusteringAlgoInterface which performs the clustering and outputs clusters;
    then self decides on the cluster-neurite correspondence and assigns in self.annotations.
    """
    def __init__(self, data, parameters, kind=None, *args, **kwargs):
        """
        Sets up a clustering object with given data and algorithm.
        :param data: instance of Dataset
        :param parameters: instance of Parameters with kind that contains "cluster"
        :param kind: optional; None or one of "graph", "km_range"; which clustering algo is to be used.
            If None or not provided, will be determined from parameters; otherwise will override value in parameters.
        """
        # alternatively, __init__ could be without the kind parameter and could ask the user.
        super(Clustering, self).__init__(data, *args, **kwargs)

        # if necessary, determine kind based on parameters:
        if kind is None:
            kind = "graph" if parameters["graph_cluster"] else "km"
        # depending on kind, create instance of the right ClusteringAlgoInterface subclass:
        if "graph" in kind:
            self.algo = GraphClustering(data, parameters)
        elif "km" in kind.lower():
            self.algo = KmRangeClustering(data, parameters)
        else:
            raise ValueError("kind must contain one of ('graph', 'km')")

    # Todo: clean up args in prepare and find_assignment
    def prepare(self, *args, **kwargs):
        pass

    def find_assignment(self, frames, is_first=True, *args, **kwargs):
        """
        Runs the algorithms to find assignments for the segments of frames, and fills self.annotations with resulting assignments.
        :param frames: which frames to find assignments for
        :param is_first: whether this is the first time that any segments are  assigned.
            In this case, one neurite will be created per cluster found.
            Otherwise, will ask the user to choose a neurite for each cluster.
        """
        clusters_assignments = self.algo.cluster(frames)# MB: forms an object of cluster,I think it is a dictionary of segs_list and clusters
        self.assign_to_neurites(clusters_assignments, is_first)#MB: Todo: the is_first should change from inputs of the gui
        print("find_assignment Finished")

    def assign_to_neurites(self, clusters_assignments, is_first):
        """
        Choose neurite for each cluster in clusters_assignments,
        and assign (t, s)s to neurites accordingly in self.annotations.
        :param clusters_assignments: output of self.algo.cluster(). A dict (t, s) -> cluster.
        """
        if is_first:
            self.annotations.assign(clusters_assignments, update_nb_neurons=True)#MB: I think self.annotations is just the dataset
        else:
            # Todo
            raise NotImplementedError("Sorry, I didn't think you would need this feature. Feel free to implement, it shouldn't be too hard.")
        print("assigne_to_neu Finished")

class ClusteringAlgoInterface:
    """
    Interface for different clustering algorithms, invisible to main controller.
    Subclasses are only instantiated by Clustering.
    """
    # Todo: cluster could be called in __init__, or by the __init__ of Clustering?
    def __init__(self, features, params):
        self.features = features
        self.params = params
        self.segs_list = []   # todo: is it right for this to be a variable of the class?

    @abc.abstractmethod
    def cluster(self, frames):
        """
        Cluster and return cluster assignments.
        :param frames: which frames to cluster
        :return: dictionary (t, s) -> cluster of cluster assigned to each segment
        """
        raise NotImplementedError

    def prepare_features(self, frames):
        pc_var = self.params["pc_var"]
        if pc_var == "None":
            pc_var = None
        rotation_invariant = self.params["rotation_invariant"]
        further_alignment = self.params["further_alignments"]
        ftr_arr, self.segs_list = self.features.feature_array(times=frames, rotation_invariant=rotation_invariant, segs_list=True, further_alignments=further_alignment)#features for each (t,s) doublet
        scaled_ftr = scale(ftr_arr)

        if pc_var is not None:
            warnings.warn("PCA analysis may not be working right, if you don't want PCA please enter None in pc value", stacklevel=2)

            pcfit = PCA().fit(scaled_ftr)
            cumsum = np.cumsum(pcfit.explained_variance_ratio_)

            npc = np.argwhere(cumsum > pc_var)[0, 0]
            cluster_features = pcfit.transform(scaled_ftr)[:, :npc - 1]
        else:
            cluster_features = scaled_ftr
        return cluster_features#MB: a vector of features for each (t,s)


class KmRangeClustering(ClusteringAlgoInterface):
    """Perform K-Means clustering over a range of number of clusters."""
    def __init__(self, features, params):
        super(KmRangeClustering, self).__init__(features, params)
        self.logger = logging.getLogger("KmCluster")

    def cluster(self, frames):
        self.logger.debug('Clustering frames {} with parameters {}'.format(frames, self.params))
        km_dict = h.timed_func("km range clustering")(self.km_range)(frames)
        nb_clusters = int(input("number of clusters"))   # TODO: better interface!!
        return dict(zip(self.segs_list, km_dict[nb_clusters]))#a dictionary of two lists

    def km_range(self, frames, plot=True):
        """
        Performs k means clustering over a range of cluster numbers.
        :param frames: which frames to cluster
        :param plot:   If set to true, creates inertia plot
        :return:       List of cluster assignments (dictionary of a nember of clusters and list of labels)

        TODO pca analysis seems to mess up when there are few

        """
        cluster_features = self.prepare_features(frames)  # here not in do_clustering so PCA is included in timing

        km_dict = {}   # dict nb_clusters: KMeans object with nb_clusters clusters
        clrange = self.params["clrange"]
        for ncl in range(clrange[0], clrange[1]):
            if ncl < cluster_features.shape[0]:  # Can't try to cluster when there aren't even any elements(MB: number of clusters should be less than number of (t,s)which are the datat here)
                km = KMeans(n_clusters=ncl)
                km.fit(cluster_features)#MB: I think this line clusters the data matrix(feature matrix here)
                km_dict[ncl] = km# for each nb_cluster we have one cluster object

        if plot:
            plt.plot([c.inertia_ for c in km_dict.values()])#the smaller the inertia, the denser the cluster and the closer the points are
            plt.title("Inertia vs Clustering Ix")
            plt.show()

        k_cluster_dict = {ncl: x.labels_ for ncl, x in km_dict.items()}
        return k_cluster_dict


class GraphClustering(ClusteringAlgoInterface):
    """Perform Matthias' graph-based clustering"""
    def __init__(self, features, params):
        super(GraphClustering, self).__init__(features, params)
        self.logger = logging.getLogger("GraphCluster")

    def cluster(self, frames):
        self.logger.debug('Clustering frames {} with parameters {}'.format(frames, self.params))
        cluster_features = self.prepare_features(frames)
        print(frames)
        time = frames#self.features.feature_times()
        n_neighbours = self.params["graph_nneighbors"]
        clusters = graph_based_cluster(cluster_features, time, nneighbors=n_neighbours, verbose=False)
        print("GraphClustering finished")
        return dict(zip(self.segs_list, clusters))
