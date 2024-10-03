# For evaluation an appropriate number of clusters 
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import pandas as pd
import os



class OptimalClusterFinder:
    """
    A class for determining the optimal number of clusters in a dataset using various methods.

    Attributes:
        __kmeans_init (str): Initialization method for KMeans clustering algorithm.
        __kmeans_random_state (int): Random state for reproducibility in KMeans.
        __gauss_random_state (int): Random state for reproducibility in Gaussian Mixture Model.
        __dpi (int): Dots per inch for saving high-quality graphs.
    """
    
    def __init__(self, kmeans_init: str, kmeans_random_state: int, gauss_random_state: int, dpi: int):
        """
        Initializes the OptimalClusterFinder object with KMeans initialization method and random state.

        Args:
            kmeans_init (str): Initialization method for KMeans clustering algorithm (e.g., 'k-means++', 'random').
            kmeans_random_state (int): Random state for reproducibility of KMeans results.
            gauss_random_state (int): Random state for reproducibility of Gaussian Mixture Model results.
            dpi (int): Dots per inch setting for saving high-quality graphs.
        """
        self.__kmeans_init = kmeans_init
        self.__kmeans_random_state = kmeans_random_state
        self.__gauss_random_state = gauss_random_state
        self.__dpi = dpi

    def k_elbow_method(self, dataset: pd.DataFrame, folder_path: str, k_elbow_path: str, 
                       file_name: str) -> None:
        """
        Applies the Elbow method to find the optimal number of clusters and saves the plot.

        Args:
            dataset (pd.DataFrame): The dataset for clustering.
            folder_path (str): The path to save the elbow method plot.
            k_elbow_path (str): Subdirectory for the elbow method plot.
            file_name (str): Name of the file to save the plot.
        """
        full_path = os.path.join(folder_path, k_elbow_path)
        os.makedirs(full_path, exist_ok=True)

        model = KMeans(init=self.__kmeans_init, random_state=self.__kmeans_random_state)
        visualizer = KElbowVisualizer(model, k=(1,25), timings=False)
        visualizer.fit(dataset)
        optimal_k = visualizer.elbow_value_

        visualizer.ax.set_xlabel('Cluster number (k)')
        visualizer.ax.set_ylabel('WCSS')
        visualizer.ax.set_title('The Elbow Method')
        visualizer.ax.text(
            0.95, 0.95,
            f'The optimal number of clusters: {optimal_k}', 
            ha='right', va='top',
            fontsize=12, color='blue',
            transform=visualizer.ax.transAxes
        )
        
        path = f"{os.path.join(full_path, file_name)}.png"
        visualizer.ax.figure.savefig(path, dpi = self.__dpi)
        plt.close()

    def silhouette_score_method(self, dataset: pd.DataFrame, folder_path: str, 
                                silhouette_score_path: str, file_name: str) -> None:
        """
        Computes and plots silhouette scores for a range of cluster numbers and saves the plots.

        Args:
            dataset (pd.DataFrame): The dataset for clustering.
            folder_path (str): The path to save the silhouette score plots.
            silhouette_score_path (str): Subdirectory for the silhouette score plots.
            file_name (str): Name of the files to save the plots.
        """
        full_path = os.path.join(folder_path, silhouette_score_path)
        os.makedirs(full_path, exist_ok=True)
        
        silhouette_scores = []
        range_n_clusters = range(2, 15)
        for _, n_clusters in enumerate(range_n_clusters):
            clusterer = KMeans(n_clusters=n_clusters, init=self.__kmeans_init, random_state=self.__kmeans_random_state)

            cluster_labels = clusterer.fit_predict(dataset)
            silhouette_avg = silhouette_score(dataset, cluster_labels)
            silhouette_scores.append(silhouette_avg)

            visualizer = SilhouetteVisualizer(clusterer, colors='yellowbrick')
            visualizer.fit(dataset)
            visualizer.finalize()

            path = f"{full_path}/{file_name}_{n_clusters}_clusters.png"
            visualizer.show(outpath=path, dpi = self.__dpi)
            plt.close()

        plt.figure(figsize=(8, 6))
        plt.plot(range_n_clusters, silhouette_scores, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score for different number of clusters')
        path = f"{full_path}/{file_name}_silhouette_average.png"
        plt.savefig(path, dpi = self.__dpi)
        plt.close()

    def dendrogram_method(self, dataset: pd.DataFrame, folder_path: str, 
                          dendrogram_path: str, file_name: str, threshold: int) -> None:
        """
        Creates a dendrogram to visualize hierarchical clustering and saves the plot.

        Args:
            dataset (pd.DataFrame): The dataset for clustering.
            folder_path (str): The path to save the dendrogram plot.
            dendrogram_path (str): Subdirectory for the dendrogram plot.
            file_name (str): Name of the file to save the plot.
            threshold (int): The threshold value to indicate clusters.
        """
        full_path = os.path.join(folder_path, dendrogram_path)
        os.makedirs(full_path, exist_ok=True)
        
        sch.dendrogram(sch.linkage(dataset, method='ward'))
        plt.axhline(y=threshold, color='black', linestyle='--')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.title('Dendrogram')

        path=f"{os.path.join(full_path, file_name)}.png"
        plt.savefig(path, dpi = self.__dpi)
        plt.close()

    def bic_aic_for_gaussian_method(self, dataset: pd.DataFrame, folder_path: str, 
                                bic_aic_path: str, file_name: str) -> None:
        """
        Computes and plots BIC and AIC scores for different numbers of components in a Gaussian Mixture Model.

        Args:
            dataset (pd.DataFrame): The dataset for clustering.
            folder_path (str): The path to save the BIC and AIC plots.
            bic_aic_path (str): Subdirectory for the BIC and AIC plots.
            file_name (str): Name of the file to save the plots.
        """
        full_path = os.path.join(folder_path, bic_aic_path)
        os.makedirs(full_path, exist_ok=True)

        n_components_range = range(1, 15)
        bic_scores = []
        aic_scores = []

        for n_components in n_components_range:
            gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=self.__gauss_random_state)
            gmm.fit(dataset)            
            bic_scores.append(gmm.bic(dataset))
            aic_scores.append(gmm.aic(dataset))

        plt.figure(figsize=(8, 6))
        plt.plot(n_components_range, bic_scores, label='BIC', marker='o')
        plt.plot(n_components_range, aic_scores, label='AIC', marker='o')
        plt.xlabel('Number of components')
        plt.ylabel('Score')
        plt.title('BIC and AIC for Gaussian Mixture Model')
        plt.legend()

        path=f"{os.path.join(full_path, file_name)}.png"
        plt.savefig(path, dpi = self.__dpi)
        plt.close()
