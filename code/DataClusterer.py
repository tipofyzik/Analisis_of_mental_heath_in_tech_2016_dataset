from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

import numpy as np

class DataClusterer:
    def __init__(self, cluster_number: int, kmeans_random_state: int, kmeans_init: str, gauss_random_state: int):
        self.__cluster_number = cluster_number
        self.__gauss_random_state = gauss_random_state
        self.__kmeans_random_state = kmeans_random_state
        self.__kmeans_init = kmeans_init

    def gaussian_mixture_clusterization(self, reduced_dataset: np.ndarray):
        final_gmm = GaussianMixture(n_components=self.__cluster_number, random_state=self.__gauss_random_state)
        gauss_cluster_labels = final_gmm.fit_predict(reduced_dataset)
        return gauss_cluster_labels

    def kmeans_clusterizaiton(self, reduced_dataset: np.ndarray):
        kmeans = KMeans(n_clusters = self.__cluster_number, init=self.__kmeans_init, random_state=self.__kmeans_random_state)
        kmeans_cluster_labels = kmeans.fit_predict(reduced_dataset)
        return kmeans_cluster_labels

    def agglomerative_clusterization(self, reduced_dataset: np.ndarray):
        clustering = AgglomerativeClustering(n_clusters=self.__cluster_number, metric='euclidean', linkage='ward')
        agglomerative_cluster_labels = clustering.fit_predict(reduced_dataset)
        return agglomerative_cluster_labels
    