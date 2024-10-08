# Linear dimensionality reduction
from sklearn.decomposition import PCA

# Non-linear dimensionality reduction
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE, MDS
from sklearn.metrics import pairwise_distances

import pandas as pd
import numpy as np



class DimensionalityReducer:
    """
    A class to perform dimensionality reduction using various techniques such as PCA, Kernel PCA, t-SNE, and MDS.

    Attributes:
        __pca_2d: PCA object for reducing dimensionality to 2D.
        __pca_3d: PCA object for reducing dimensionality to 3D.
        __kpca_2d: KernelPCA object for reducing dimensionality to 2D using RBF kernel.
        __kpca_3d: KernelPCA object for reducing dimensionality to 3D using RBF kernel.
        __tsne_2d: TSNE object for reducing dimensionality to 2D.
        __tsne_3d: TSNE object for reducing dimensionality to 3D.
        __mds_2d: MDS object for reducing dimensionality to 2D.
        __mds_3d: MDS object for reducing dimensionality to 3D.
    """
    
    def __init__(self, pca_random_state: int, kpca_kernel: str, kernel_pca_random_state: int, 
                 tsne_random_state: int, mds_random_state: int) -> None:
        """
        Initializes the DimensionalityReducer with specified random states for various algorithms.

        Args:
            pca_random_state (int): Seed for the random number generator used in PCA.
            kpca_kernel (str): Kernel function name used in Kernel PCA.
            kernel_pca_random_state (int): Seed for the random number generator used in Kernel PCA.
            tsne_random_state (int): Seed for the random number generator used in t-SNE.
            mds_random_state (int): Seed for the random number generator used in MDS.

        The instance will initialize PCA, Kernel PCA, t-SNE, and MDS objects with 2D and 3D 
        configurations, using the specified random states.
        """
        self.__pca_2d = PCA(n_components = 2, random_state = pca_random_state)
        self.__pca_3d = PCA(n_components = 3, random_state = pca_random_state)
        self.__kpca_2d = KernelPCA(n_components=2, random_state = kernel_pca_random_state, kernel=kpca_kernel)
        self.__kpca_3d = KernelPCA(n_components=3, random_state = kernel_pca_random_state, kernel=kpca_kernel)
        self.__tsne_2d = TSNE(n_components=2, random_state=tsne_random_state)
        self.__tsne_3d = TSNE(n_components=3, random_state=tsne_random_state)
        self.__mds_2d = MDS(n_components=2, random_state=mds_random_state, dissimilarity='precomputed')
        self.__mds_3d = MDS(n_components=3, random_state=mds_random_state, dissimilarity='precomputed')

    def __pca_variance(self, saved_info_ratio: float, pca_random_state: int, 
                       normalized_dataset: pd.DataFrame) -> None:
        """
        Calculates and prints PCA explained variance ratios and cumulative variance.

        Args:
            saved_info_ratio (float): The target cumulative explained variance ratio to retain.
            pca_random_state (int): Seed for the random number generator used in PCA.
            normalized_dataset (pd.DataFrame): The dataset to apply PCA on, which should be normalized.
        """
        self.__pca = PCA(n_components = saved_info_ratio, random_state = pca_random_state)
        self.__pca_result = self.__pca.fit_transform(normalized_dataset)

        # Cheching explaing variance
        explained_variance_ratio = self.__pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)

        # Output of results
        print(f"Explained variance ratio of each component: {explained_variance_ratio}")
        print(f"Cumulative explained variance: {cumulative_variance}")
        print(np.sum(self.__pca.explained_variance_ratio_))

    def get_n_percent_pca_result(self, saved_info_ratio: float, pca_random_state: int, 
                                 normalized_dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Performs PCA on the normalized dataset, preserving the specified explained variance ratio.

        Args:
            saved_info_ratio (float): The target cumulative explained variance ratio to retain.
            pca_random_state (int): Seed for the random number generator used in PCA.
            normalized_dataset (pd.DataFrame): The normalized dataset to be transformed using PCA.

        Returns:
            pd.DataFrame: The PCA-transformed dataset, retaining the components that contribute 
            to the specified explained variance ratio.
        """
        self.__pca_variance(saved_info_ratio, pca_random_state, normalized_dataset)
        return self.__pca_result

    def __linear_pca_2d_reducer(self, normalized_dataset: pd.DataFrame) -> None:
        """
        Reduces the dimensionality of the normalized dataset to 2D using linear PCA.

        Args:
            normalized_dataset (pd.DataFrame): The normalized dataset for PCA.
        """
        self.__pca_2d_result = self.__pca_2d.fit_transform(normalized_dataset)

    def __linear_pca_3d_reducer(self, normalized_dataset: pd.DataFrame) -> None:
        """
        Reduces the dimensionality of the normalized dataset to 3D using linear PCA.

        Args:
            normalized_dataset (pd.DataFrame): The normalized dataset for PCA.
        """
        self.__pca_3d_result = self.__pca_3d.fit_transform(normalized_dataset)
    
    def __kernel_pca_2d_reducer(self, normalized_dataset: pd.DataFrame) -> None:
        """
        Reduces the dimensionality of the normalized dataset to 2D using kernel PCA.

        Args:
            normalized_dataset (pd.DataFrame): The normalized dataset for kernel PCA.
        """
        self.__kpca_2d_result = self.__kpca_2d.fit_transform(normalized_dataset)

    def __kernel_pca_3d_reducer(self, normalized_dataset: pd.DataFrame) -> None:
        """
        Reduces the dimensionality of the normalized dataset to 3D using kernel PCA.

        Args:
            normalized_dataset (pd.DataFrame): The normalized dataset for kernel PCA.
        """
        self.__kpca_3d_result = self.__kpca_3d.fit_transform(normalized_dataset)

    def __tsne_2d_reducer(self, normalized_or_pca_dataset: pd.DataFrame) -> None:
        """
        Reduces the dimensionality of the dataset to 2D using t-SNE.

        Args:
            normalized_or_pca_dataset (pd.DataFrame): The dataset for t-SNE, which can be normalized or PCA-transformed.
        """
        self.__tsne_2d_result = self.__tsne_2d.fit_transform(normalized_or_pca_dataset)


    def __tsne_3d_reducer(self, normalized_or_pca_dataset: pd.DataFrame) -> None:
        """
        Reduces the dimensionality of the dataset to 3D using t-SNE.

        Args:
            normalized_or_pca_dataset (pd.DataFrame): The dataset for t-SNE, which can be normalized or PCA-transformed.
        """
        self.__tsne_3d_result = self.__tsne_3d.fit_transform(normalized_or_pca_dataset)

    def __mds_2d_reducer(self, encoded_dataset: pd.DataFrame) -> None:
        """
        Reduces the dimensionality of the dataset to 2D using MDS.

        Args:
            encoded_dataset (pd.DataFrame): The dataset for MDS, typically 
            consisting of categorical data that has been one-hot encoded or 
            otherwise transformed into a suitable format.
        """
        distance_matrix = pairwise_distances(encoded_dataset, metric='hamming')
        self.__mds_2d_result = self.__mds_2d.fit_transform(distance_matrix)

    def __mds_3d_reducer(self, encoded_dataset: pd.DataFrame) -> None:
        """
        Reduces the dimensionality of the dataset to 3D using MDS.

        Args:
            encoded_dataset (pd.DataFrame): The dataset for MDS, typically 
            consisting of categorical data that has been one-hot encoded or 
            otherwise transformed into a suitable format.
        """        
        distance_matrix = pairwise_distances(encoded_dataset, metric='hamming')
        self.__mds_3d_result = self.__mds_3d.fit_transform(distance_matrix)

    def get_linear_pca_result(self, normalized_dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Applies linear PCA to the provided normalized dataset, computes the reduced datasets 
        in both 2D and 3D dimensions, and returns them.

        Args:
            normalized_dataset (pd.DataFrame): The normalized dataset for PCA.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the PCA-transformed datasets 
        for 2D and 3D dimensionality reduction, respectively.
        """
        self.__linear_pca_2d_reducer(normalized_dataset)
        self.__linear_pca_3d_reducer(normalized_dataset)
        return self.__pca_2d_result, self.__pca_3d_result
    

    def get_kernel_pca_result(self, normalized_or_pca_dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Applies kernel PCA to the provided dataset and computes the reduced datasets 
        in both 2D and 3D dimensions, returning them.

        Args:
            normalized_or_pca_dataset (pd.DataFrame): The dataset for kernel PCA, which can be either normalized 
            or PCA-transformed.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the kernel PCA-transformed datasets 
            for 2D and 3D dimensionality reduction, respectively.
        """
        self.__kernel_pca_2d_reducer(normalized_or_pca_dataset)
        self.__kernel_pca_3d_reducer(normalized_or_pca_dataset)
        return self.__kpca_2d_result, self.__kpca_3d_result
    
    def get_tsne_result(self, normalized_or_pca_dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Applies t-SNE to the provided dataset and computes the reduced datasets 
        in both 2D and 3D dimensions, returning them.

        Args:
            normalized_or_pca_dataset (pd.DataFrame): The dataset for t-SNE, which can be either normalized 
            or PCA-transformed.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the t-SNE-transformed datasets 
            for 2D and 3D dimensionality reduction, respectively.
        """
        self.__tsne_2d_reducer(normalized_or_pca_dataset)
        self.__tsne_3d_reducer(normalized_or_pca_dataset)
        return self.__tsne_2d_result, self.__tsne_3d_result
    
    def get_mds_result(self, encoded_dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Applies MDS to the provided dataset and computes the reduced datasets 
        in both 2D and 3D dimensions, returning them.

        Args:
            encoded_dataset (pd.DataFrame): The dataset for MDS, typically consisting of categorical data 
            that has been one-hot encoded or transformed into a suitable format for analysis.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the MDS-transformed datasets 
            for 2D and 3D dimensionality reduction, respectively.
        """
        self.__mds_2d_reducer(encoded_dataset)
        self.__mds_3d_reducer(encoded_dataset)
        return self.__mds_2d_result, self.__mds_3d_result 