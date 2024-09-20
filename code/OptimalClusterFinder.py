# For evaluation an appropriate number of clusters 
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
import scipy.cluster.hierarchy as sch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

class OptimalClusterFinder:
    def __init__(self, kmeans_init: str, random_state: int):
        self.__kmeans_init = kmeans_init
        self.__random_state = random_state

    def k_elbow_method(self, dataset: pd.DataFrame, folder_path: str, k_elbow_path: str, 
                       file_name: str):
        full_path = os.path.join(folder_path, k_elbow_path)
        os.makedirs(full_path, exist_ok=True)

        model = KMeans(init=self.__kmeans_init, random_state=self.__random_state)
        visualizer = KElbowVisualizer(model, k=(1,25), timings=False)
        # fit the visualizer and show the plot
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
        visualizer.ax.figure.savefig(path, dpi=300)
        plt.close()

    def silhouette_score_method(self, dataset: pd.DataFrame, folder_path: str, 
                                silhouette_score_path: str, file_name: str):
        full_path = os.path.join(folder_path, silhouette_score_path)
        os.makedirs(full_path, exist_ok=True)

        range_n_clusters = range(2, 15)
        for _, n_clusters in enumerate(range_n_clusters):
            # Создайте новую фигуру и ось для каждого графика
            fig, ax = plt.subplots(figsize=(8, 5))
            
            # Кластеризация и расчет коэффициентов силуэта
            clusterer = KMeans(n_clusters=n_clusters, init=self.__kmeans_init, random_state=self.__random_state)
            cluster_labels = clusterer.fit_predict(dataset)
            silhouette_avg = silhouette_score(dataset, cluster_labels)
            sample_silhouette_values = silhouette_samples(dataset, cluster_labels)
            
            y_lower = 10
            
            for j in range(n_clusters):
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == j]
                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = plt.cm.viridis(float(j) / n_clusters)
                ax.fill_betweenx(np.arange(y_lower, y_upper),
                                0, ith_cluster_silhouette_values,
                                facecolor=color, edgecolor='k', alpha=0.6)
                
                y_lower = y_upper + 10  # Отступ между кластерами

            ax.set_title(f"Silhouette Plot for {n_clusters} Clusters", fontsize=12)
            ax.set_xlabel("Silhouette Coefficient Values", fontsize=10)
            ax.set_ylabel("Cluster Label", fontsize=10)
            ax.axvline(x=silhouette_avg, color="red", linestyle="--")
            ax.set_yticks([])
            ax.set_xticks(np.linspace(-0.1, 1.0, 11))
            ax.grid(True, linestyle='--', alpha=0.7)

            path = f"{full_path}/{file_name}_{n_clusters}_clusters.png"
            fig.savefig(path, dpi=300)
            plt.close()

    def dendrogram_method(self, dataset: pd.DataFrame, folder_path: str, 
                                dendrogram_path: str, file_name: str,
                                threshold: int):
        full_path = os.path.join(folder_path, dendrogram_path)
        os.makedirs(full_path, exist_ok=True)
        
        dendrogram = sch.dendrogram(sch.linkage(dataset, method='ward'))
        plt.axhline(y=threshold, color='black', linestyle='--')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.title('Dendrogram')

        path=f"{os.path.join(full_path, file_name)}.png"
        plt.savefig(path, dpi=300)  # Сохранение с разрешением 300 dpi
        plt.close()