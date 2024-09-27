from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

import numpy as np

class DataClusterer:
    """
    A class to perform various clustering algorithms on a reduced dataset,
    including KMeans, Gaussian Mixture Models, and Agglomerative Clustering.

    Attributes:
        __cluster_number: The number of clusters to form.
        __gauss_random_state: The random state for Gaussian Mixture.
        __kmeans_random_state: The random state for KMeans.
        __kmeans_init: The initialization method for KMeans.
    """

    def __init__(self, cluster_number: int, kmeans_random_state: int, kmeans_init: str, gauss_random_state: int):
        """
        Initializes the DataClusterer with the specified parameters.

        Args:
            cluster_number (int): The number of clusters to form.
            kmeans_random_state (int): The random state for KMeans.
            kmeans_init (str): The initialization method for KMeans.
            gauss_random_state (int): The random state for Gaussian Mixture.
        """
        self.__cluster_number = cluster_number
        self.__gauss_random_state = gauss_random_state
        self.__kmeans_random_state = kmeans_random_state
        self.__kmeans_init = kmeans_init

    def gaussian_mixture_clusterization(self, reduced_dataset: np.ndarray) -> np.ndarray:
        """
        Applies Gaussian Mixture clustering on the reduced dataset.

        Args:
            reduced_dataset (np.ndarray): The dataset to cluster.

        Returns:
            np.ndarray: The cluster labels assigned by the Gaussian Mixture model.
        """
        final_gmm = GaussianMixture(n_components=self.__cluster_number, random_state=self.__gauss_random_state)
        gauss_cluster_labels = final_gmm.fit_predict(reduced_dataset)
        return gauss_cluster_labels

    def kmeans_clusterization(self, reduced_dataset: np.ndarray) -> np.ndarray:
        """
        Applies KMeans clustering on the reduced dataset.

        Args:
            reduced_dataset (np.ndarray): The dataset to cluster.

        Returns:
            np.ndarray: The cluster labels assigned by KMeans.
        """
        kmeans = KMeans(n_clusters = self.__cluster_number, init=self.__kmeans_init, random_state=self.__kmeans_random_state)
        kmeans_cluster_labels = kmeans.fit_predict(reduced_dataset)
        return kmeans_cluster_labels

    def agglomerative_clusterization(self, reduced_dataset: np.ndarray) -> np.ndarray:
        """
        Applies Agglomerative Clustering on the reduced dataset.

        Args:
            reduced_dataset (np.ndarray): The dataset to cluster.

        Returns:
            np.ndarray: The cluster labels assigned by Agglomerative Clustering.
        """
        clustering = AgglomerativeClustering(n_clusters=self.__cluster_number, metric='euclidean', linkage='ward')
        agglomerative_cluster_labels = clustering.fit_predict(reduced_dataset)
        return agglomerative_cluster_labels
    