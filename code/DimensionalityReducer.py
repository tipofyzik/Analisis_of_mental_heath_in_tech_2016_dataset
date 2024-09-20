# Linear dimensionality reduction
from sklearn.decomposition import PCA

# Non-linear dimensionality reduction
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE

import pandas as pd
import numpy as np

class DimensionalityReducer:
    def __init__(self, random_state: int):
        self.__pca_2d = PCA(n_components = 2)
        self.__pca_3d = PCA(n_components = 3)
        self.__kpca_2d = KernelPCA(n_components=2, kernel='rbf')
        self.__kpca_3d = KernelPCA(n_components=3, kernel='rbf')
        self.__tsne_2d = TSNE(n_components=2, random_state=random_state)
        self.__tsne_3d = TSNE(n_components=3, random_state=random_state)


    def __pca_variance(self, saved_info_ratio: float, normalized_dataset: pd.DataFrame):
        self.__pca = PCA(n_components = saved_info_ratio)
        self.__pca_result = self.__pca.fit_transform(normalized_dataset)

        # Проверка объясненной дисперсии
        explained_variance_ratio = self.__pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)

        # Вывод результатов
        print(f"Explained variance ratio of each component: {explained_variance_ratio}")
        print(f"Cumulative explained variance: {cumulative_variance}")
        print(np.sum(self.__pca.explained_variance_ratio_))

    def get_95_percent_pca_result(self, saved_info_ratio: float, normalized_dataset: pd.DataFrame):
        self.__pca_variance(saved_info_ratio, normalized_dataset)
        return self.__pca_result

    def __linear_pca_2d_reducer(self, normalized_dataset: pd.DataFrame):
        self.__pca_2d_result = self.__pca_2d.fit_transform(normalized_dataset)

    def __linear_pca_3d_reducer(self, normalized_dataset: pd.DataFrame):
        self.__pca_3d_result = self.__pca_3d.fit_transform(normalized_dataset)
    
    def __kernel_pca_2d_reducer(self, normalized_dataset: pd.DataFrame):
        self.__kpca_2d_result = self.__kpca_2d.fit_transform(normalized_dataset)

    def __kernel_pca_3d_reducer(self, normalized_dataset: pd.DataFrame):
        self.__kpca_3d_result = self.__kpca_3d.fit_transform(normalized_dataset)

    def __tsne_2d_reducer(self, normalized_or_pca_dataset: pd.DataFrame):
        self.__tsne_2d_result = self.__tsne_2d.fit_transform(normalized_or_pca_dataset)

    def __tsne_3d_reducer(self, normalized_or_pca_dataset: pd.DataFrame):
        self.__tsne_3d_result = self.__tsne_3d.fit_transform(normalized_or_pca_dataset)

    def get_linear_pca_result(self, normalized_dataset: pd.DataFrame):
        self.__linear_pca_2d_reducer(normalized_dataset)
        self.__linear_pca_3d_reducer(normalized_dataset)
        return self.__pca_2d_result, self.__pca_3d_result
    
    def get_kernel_pca_result(self, normalized_or_pca_dataset: pd.DataFrame):
        self.__kernel_pca_2d_reducer(normalized_or_pca_dataset)
        self.__kernel_pca_3d_reducer(normalized_or_pca_dataset)
        return self.__kpca_2d_result, self.__kpca_3d_result
    
    def get_tsne_result(self, normalized_or_pca_dataset: pd.DataFrame):
        self.__tsne_2d_reducer(normalized_or_pca_dataset)
        self.__tsne_3d_reducer(normalized_or_pca_dataset)
        return self.__tsne_2d_result, self.__tsne_3d_result
    