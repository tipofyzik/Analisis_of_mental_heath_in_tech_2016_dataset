# Linear dimensionality reduction
from sklearn.decomposition import PCA

# Non-linear dimensionality reduction
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE

import pandas as pd
import numpy as np

class DimensionalityReducer:
    """
    A class to perform dimensionality reduction using various techniques such as PCA, Kernel PCA, and t-SNE.

    Attributes:
        __pca_2d: PCA object for 2D dimensionality reduction.
        __pca_3d: PCA object for 3D dimensionality reduction.
        __kpca_2d: KernelPCA object for 2D dimensionality reduction using RBF kernel.
        __kpca_3d: KernelPCA object for 3D dimensionality reduction using RBF kernel.
        __tsne_2d: TSNE object for 2D dimensionality reduction.
        __tsne_3d: TSNE object for 3D dimensionality reduction.
    """
    
    def __init__(self, random_state: int) -> None:
        """
        Initializes the DimensionalityReducer with specified random state.

        Args:
            random_state (int): Seed for random number generator.
            # tsne_perplexity (float): The perplexity parameter is related to the number of nearest neighbors.
        """
        self.__pca_2d = PCA(n_components = 2)
        self.__pca_3d = PCA(n_components = 3)
        self.__kpca_2d = KernelPCA(n_components=2, kernel='rbf')
        self.__kpca_3d = KernelPCA(n_components=3, kernel='rbf')
        self.__tsne_2d = TSNE(n_components=2, random_state=random_state)
        self.__tsne_3d = TSNE(n_components=3, random_state=random_state)

    def __pca_variance(self, saved_info_ratio: float, normalized_dataset: pd.DataFrame) -> None:
        """
        Calculates PCA variance and prints explained variance ratios and cumulative variance.

        Args:
            saved_info_ratio (float): The desired explained variance ratio to keep.
            normalized_dataset (pd.DataFrame): The normalized dataset for PCA.
        """
        self.__pca = PCA(n_components = saved_info_ratio)
        self.__pca_result = self.__pca.fit_transform(normalized_dataset)

        # Cheching explaing variance
        explained_variance_ratio = self.__pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)

        # Output of results
        print(f"Explained variance ratio of each component: {explained_variance_ratio}")
        print(f"Cumulative explained variance: {cumulative_variance}")
        print(np.sum(self.__pca.explained_variance_ratio_))

    def get_95_percent_pca_result(self, saved_info_ratio: float, normalized_dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Gets the PCA result preserving the specified explained variance ratio.

        Args:
            saved_info_ratio (float): The desired explained variance ratio to keep.
            normalized_dataset (pd.DataFrame): The normalized dataset for PCA.

        Returns:
            pd.DataFrame: The PCA transformed dataset.
        """
        self.__pca_variance(saved_info_ratio, normalized_dataset)
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


    def get_linear_pca_result(self, normalized_dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Gets the PCA results for both 2D and 3D dimensionality reduction.

        Args:
            normalized_dataset (pd.DataFrame): The normalized dataset for PCA.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: The PCA transformed datasets for 2D and 3D.
        """
        self.__linear_pca_2d_reducer(normalized_dataset)
        self.__linear_pca_3d_reducer(normalized_dataset)
        return self.__pca_2d_result, self.__pca_3d_result
    

    def get_kernel_pca_result(self, normalized_or_pca_dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Gets the kernel PCA results for both 2D and 3D dimensionality reduction.

        Args:
            normalized_or_pca_dataset (pd.DataFrame): The dataset for kernel PCA, which can be normalized or PCA-transformed.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: The kernel PCA transformed datasets for 2D and 3D.
        """
        self.__kernel_pca_2d_reducer(normalized_or_pca_dataset)
        self.__kernel_pca_3d_reducer(normalized_or_pca_dataset)
        return self.__kpca_2d_result, self.__kpca_3d_result
    
    def get_tsne_result(self, normalized_or_pca_dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Gets the t-SNE results for both 2D and 3D dimensionality reduction.

        Args:
            normalized_or_pca_dataset (pd.DataFrame): The dataset for t-SNE, which can be normalized or PCA-transformed.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: The t-SNE transformed datasets for 2D and 3D.
        """
        self.__tsne_2d_reducer(normalized_or_pca_dataset)
        self.__tsne_3d_reducer(normalized_or_pca_dataset)
        return self.__tsne_2d_result, self.__tsne_3d_result
    