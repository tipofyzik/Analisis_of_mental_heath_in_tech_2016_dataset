import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os



class GraphPlotter:
    """
    A class for creating various types of plots for data visualization. 

    Attributes:
        __n_columns (int): Number of columns in the grid for plotting.
        __n_rows (int): Number of rows in the grid for plotting.
        __max_feature_number (int): Maximum number of features to display in a plot.
        __dpi (int): Dots per inch setting for saving high-quality graphs.
    """
    
    def __init__(self, n_columns: int, n_rows: int, max_feature_number: int, dpi: int):
        """
        Initializes the GraphPlotter object with the number of columns, rows, and max features.

        Args:
            n_columns (int): Number of columns in the grid for plotting.
            n_rows (int): Number of rows in the grid for plotting.
            max_feature_number (int): Maximum number of features to display in a plot.
            dpi (int): Dots per inch setting for saving high-quality graphs.
        """
        self.__n_columns = n_columns
        self.__n_rows = n_rows
        self.__max_feature_number = max_feature_number
        self.__dpi = dpi

    def __display_data_per_question(self, dataset: pd.DataFrame, column_index: int, ax) -> None:
        """
        Displays a count plot for a specific column in the dataset.

        Args:
            dataset (pd.DataFrame): The dataset containing the data to plot.
            column_index (int): Index of the column to plot.
            ax (Any): Axes object to plot on.
        """
        column = dataset.columns[column_index]
        top_n_answers = dataset[column].value_counts().nlargest(self.__max_feature_number).index
        sns.countplot(y=column, data=dataset[dataset[column].isin(top_n_answers)], 
                      ax=ax, order=top_n_answers, palette='magma')
    
        ax.set_xlabel('Count', fontsize=9)
        ax.set_ylabel('Answers', fontsize=9)
        ax.tick_params(axis='x', labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
        ax.text(0.95, 0.05, column, horizontalalignment='right', verticalalignment='bottom', 
            transform=ax.transAxes, fontsize=8, color='black', style='normal')

    def __plot_columns_in_batches(self, dataset: pd.DataFrame, start_idx: int, end_idx: int, 
                                   filename: str, function: callable) -> None:
        """
        Plots columns in batches and saves the figure.

        Args:
            dataset (pd.DataFrame): The dataset for plotting.
            start_idx (int): Starting index for the batch of columns.
            end_idx (int): Ending index for the batch of columns.
            filename (str): Filename to save the plot.
            function (callable): Function to call for plotting each column.
        """
        n_columns = self.__n_columns  # Number of columns in the grid
        n_rows = self.__n_rows  # Number of rows in the grid
        n_plots = n_columns * n_rows

        # Create a new figure for each batch
        fig, axes = plt.subplots(n_rows, n_columns, figsize=(25, 10))
        axes = axes.flatten()
        for i in range(start_idx, min(end_idx, start_idx + n_plots)):
            function(dataset, i, axes[i - start_idx])
        for j in range(end_idx - start_idx, n_plots):
            axes[j].axis('off')

        plt.tight_layout()
        plt.savefig(filename, dpi = self.__dpi)
        plt.close(fig)  # Close the figure to free up memory

    def save_plots(self, path_to_save: str, dataset: pd.DataFrame) -> None:
        """
        Saves plots of all columns in the dataset.

        Args:
            path_to_save (str): Path to save the plots.
            dataset (pd.DataFrame): The dataset to plot.
        """
        os.makedirs(path_to_save, exist_ok=True)
        dataset = dataset.fillna('NaN')

        relative_path = path_to_save
        total_columns = len(dataset.columns)
        batch_size = self.__n_columns*self.__n_rows
        for start_idx in range(0, total_columns, batch_size):
            end_idx = min(start_idx + batch_size, total_columns)
            filename = f"{relative_path}/{start_idx + 1}_{end_idx}_questions_from_text_data.png"
            self.__plot_columns_in_batches(dataset, start_idx, end_idx, filename, self.__display_data_per_question)

    def save_2d_reduced_data_plotes(self, path_to_save: str, file_name: str, 
                                     reducing_method: str, reduced_data: np.ndarray) -> None:
        """
        Saves a 2D scatter plot of dimensionality-reduced data.

        Args:
            path_to_save (str): Path to save the plot.
            file_name (str): Name of the file to save the plot.
            reducing_method (str): The method used for dimensionality reduction.
            reduced_data (np.ndarray): The reduced data to plot.
        """
        os.makedirs(path_to_save, exist_ok=True)
        full_path = os.path.join(path_to_save, file_name)

        plt.figure(figsize=(10, 8))
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=reduced_data[:, 0], cmap='tab10', s=50, alpha=0.7)
        plt.title(f"{reducing_method} Visualization")
        plt.xlabel(f"{reducing_method} Component 1")
        plt.ylabel(f"{reducing_method} Component 2")
        plt.colorbar(label="Cluster Label")

        plt.tight_layout()
        plt.savefig(full_path, dpi = self.__dpi)
        plt.close()

    def save_3d_reduced_data_plotes(self, path_to_save: str, file_name: str, 
                                     reducing_method: str, reduced_data: np.ndarray,
                                     elev: int = 30, azim: int = -90) -> None:
        """
        Saves a 3D scatter plot of dimensionality-reduced data.

        Args:
            path_to_save (str): Path to save the plot.
            file_name (str): Name of the file to save the plot.
            reducing_method (str): The method used for dimensionality reduction.
            reduced_data (np.ndarray): The reduced data to plot.
            elev (int): Elevation angle for the 3D plot.
            azim (int): Azimuth angle for the 3D plot.
        """
        os.makedirs(path_to_save, exist_ok=True)
        full_path = os.path.join(path_to_save, file_name)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], c=reduced_data[:, 0], cmap='tab10')
        legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
        ax.add_artist(legend1)
        ax.set_xlabel(f"{reducing_method} Component 1")
        ax.set_ylabel(f"{reducing_method} Component 2")
        ax.set_zlabel(f"{reducing_method} Component 3")
        ax.view_init(elev=elev, azim=azim)
        plt.title(f"{reducing_method} Visualization")

        plt.tight_layout()
        plt.savefig(full_path, dpi = self.__dpi)
        plt.close()

    def save_3d_reduced_data_slice(self, path_to_save: str, file_name: str, 
                                            reducing_method: str, reduced_data: np.ndarray, 
                                            component_range: tuple[float, float] = (0, 10), component: int = 0, 
                                            angles: list[tuple[int, int]] = [(0, -90), (0, -45), (0, 0)]) -> None:
        """
        Saves a single image containing three 3D scatter plots of dimensionality-reduced data, 
        filtered by a specific range of values for a component, each plotted with different view angles.

        Args:
            path_to_save (str): Path to save the plot.
            file_name (str): Name of the file to save the plot.
            reducing_method (str): The method used for dimensionality reduction.
            reduced_data (np.ndarray): The dimensionality-reduced data to plot.
            component_range (tuple[float, float]): Range of values for filtering the component.
            component (int, optional): The component index (0, 1, or 2) to filter the data by. Default is 0.
            angles (list[tuple[int, int]], optional): List of tuples representing elevation and azimuth angles. 
                                                    Default is [(30, -90), (60, 30), (45, 120)].
        """
        os.makedirs(path_to_save, exist_ok=True)
        full_path = os.path.join(path_to_save, file_name)

        # Save the axis limits from the full dataset
        x_limits = (np.min(reduced_data[:, 0]), np.max(reduced_data[:, 0]))
        y_limits = (np.min(reduced_data[:, 1]), np.max(reduced_data[:, 1]))
        z_limits = (np.min(reduced_data[:, 2]), np.max(reduced_data[:, 2]))

        # Filter data based on the given component range
        mask = (reduced_data[:, component] >= component_range[0]) & (reduced_data[:, component] <= component_range[1])
        filtered_data = reduced_data[mask]

        _, axes = plt.subplots(1, 3, figsize=(18, 9), subplot_kw={'projection': '3d'})

        for i, (elev, azim) in enumerate(angles):
            ax = axes[i]
            ax.scatter(filtered_data[:, 0], filtered_data[:, 1], filtered_data[:, 2], c="blue")
            fontsize = 12
            ax.set_xlabel(f"{reducing_method} Component 1", fontsize = fontsize)
            ax.set_ylabel(f"{reducing_method} Component 2", fontsize = fontsize)
            ax.set_zlabel(f"{reducing_method} Component 3", fontsize = fontsize)
            ax.set_xlim(x_limits)
            ax.set_ylim(y_limits)
            ax.set_zlim(z_limits)
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f"View {i + 1}: Elev={elev}, Azim={azim}")

        # Save the figure with all three subplots
        plt.suptitle(f"{reducing_method} Visualization (Filtered by Component {component + 1})")
        plt.tight_layout()
        plt.savefig(full_path, dpi=self.__dpi)
        plt.close()

    def save_clustering_plots(self, path_to_save: str, file_name: str, type_of_clustering: str, 
                              reducing_method: str, reduced_data: np.ndarray, 
                              cluster_labels: np.ndarray) -> None:
        """
        Saves a scatter plot for clustering results.

        Args:
            path_to_save (str): Path to save the plot.
            file_name (str): Name of the file to save the plot.
            type_of_clustering (str): Type of clustering performed.
            reducing_method (str): The method used for dimensionality reduction.
            reduced_data (np.ndarray): The dimensionality-reduced data to plot.
            cluster_labels (np.ndarray): Cluster labels for the data points.
        """
        os.makedirs(path_to_save, exist_ok=True)
        full_path = os.path.join(path_to_save, file_name)

        plt.figure(figsize=(10, 8))
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels, cmap='tab10', s=50, alpha=0.7)
        plt.title(f"{type_of_clustering} Visualization")
        plt.xlabel(f"{reducing_method} Component 1")
        plt.ylabel(f"{reducing_method} Component 2")
        plt.colorbar(label="Cluster Label")

        plt.tight_layout()
        plt.savefig(full_path, dpi = self.__dpi)
        plt.close()

    def save_clustering_3d_plots(self, path_to_save: str, file_name: str, type_of_clustering: str, 
                              reducing_method: str, reduced_data: np.ndarray, 
                              cluster_labels: np.ndarray, elev: int = 30, azim: int = -90) -> None:
        """
        Saves a scatter plot for clustering results.

        Args:
            path_to_save (str): Path to the directory where the plot will be saved.
            file_name (str): Name of the file to save the plot, including the file extension (e.g., '.png').
            type_of_clustering (str): The type of clustering performed (e.g., K-means, DBSCAN).
            reducing_method (str): The method used for dimensionality reduction (e.g., PCA, t-SNE).
            reduced_data (np.ndarray): The reduced data to plot, expected to have three dimensions.
            cluster_labels (np.ndarray): Cluster labels for the data points, used for coloring.
            elev (int, optional): Elevation angle for the 3D plot. Default is 30.
            azim (int, optional): Azimuthal angle for the 3D plot. Default is -90.    
            """
        os.makedirs(path_to_save, exist_ok=True)
        full_path = os.path.join(path_to_save, file_name)

        x_limits = (np.min(reduced_data[:, 0]), np.max(reduced_data[:, 0]))
        y_limits = (np.min(reduced_data[:, 1]), np.max(reduced_data[:, 1]))
        z_limits = (np.min(reduced_data[:, 2]), np.max(reduced_data[:, 2]))
        mask = (cluster_labels == 0)
        reduced_data = reduced_data[mask]
        cluster_labels = cluster_labels[mask]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], 
                             c=cluster_labels, cmap='tab10', s=50, alpha=0.7)
        ax.set_title(f"{type_of_clustering} 3D Visualization")
        ax.set_xlabel(f"{reducing_method} Component 1")
        ax.set_ylabel(f"{reducing_method} Component 2")
        ax.set_zlabel(f"{reducing_method} Component 3")
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_zlim(z_limits)
        ax.view_init(elev=elev, azim=azim)
        fig.colorbar(scatter, ax=ax, label="Cluster Label")

        plt.tight_layout()
        plt.savefig(full_path, dpi=self.__dpi)
        plt.close()

    def plot_correlation_matrix(self, correlation_matrix: pd.DataFrame, 
                                 main_folder: str, dataset_folder: str) -> None:
        """
        Saves a heatmap of the correlation matrix.

        Args:
            correlation_matrix (pd.DataFrame): The correlation matrix to plot.
            main_folder (str): The main folder where the plot will be saved.
            dataset_folder (str): The subfolder within the main folder to save the plot.
        """
        path_to_folder = os.path.join(main_folder, dataset_folder)
        os.makedirs(path_to_folder, exist_ok=True)
        full_path = os.path.join(path_to_folder, "correlation_matrix.png")

        plt.figure(figsize=(24, 24))
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Matrix')

        plt.tight_layout()
        plt.savefig(full_path, dpi = self.__dpi)
        plt.close()

    def save_chi_squared_plot(self, chi2_result: pd.DataFrame, 
                               main_folder: str, dataset_folder: str) -> None:
        """
        Saves a bar plot of Chi-Squared feature selection results.

        Args:
            chi2_result (pd.DataFrame): The Chi-Squared results to plot.
            main_folder (str): The main folder where the plot will be saved.
            dataset_folder (str): The subfolder within the main folder to save the plot.
        """
        path_to_folder = os.path.join(main_folder, dataset_folder)
        os.makedirs(path_to_folder, exist_ok=True)
        full_path = os.path.join(path_to_folder, "chi_squared_feature_selection.png")

        # Plot Chi-Squared values
        plt.figure(figsize=(24, 8))
        sns.barplot(x='Chi-Squared', y='Feature', data=chi2_result, palette='viridis')
        plt.title('Top 20 Features by Chi-Squared Statistic')
        plt.xlabel('Chi-Squared Value')
        plt.ylabel('Feature')
        
        plt.tight_layout()
        plt.savefig(full_path, dpi = self.__dpi)
        plt.close()

    def save_mutual_info_plot(self, mutual_info_result: pd.DataFrame, 
                               main_folder: str, dataset_folder: str) -> None:
        """
        Saves a bar plot of mutual information scores.

        Args:
            mutual_info_result (pd.DataFrame): The mutual information results to plot.
            main_folder (str): The main folder where the plot will be saved.
            dataset_folder (str): The subfolder within the main folder to save the plot.
        """
        path_to_folder = os.path.join(main_folder, dataset_folder)
        os.makedirs(path_to_folder, exist_ok=True)
        full_path = os.path.join(path_to_folder, "mutual_info_feature_selection.png")
    
        # Plot mutual information scores
        plt.figure(figsize=(24, 8))
        plt.barh(mutual_info_result['Feature'], mutual_info_result['Score'], color='skyblue')
        plt.title('Mutual Information Scores of Features')
        plt.xlabel('Mutual Information Score')
        plt.ylabel('Features')
        plt.gca().invert_yaxis()

        plt.tight_layout()
        plt.savefig(full_path, dpi = self.__dpi)
        plt.close()

    def save_random_forest_plot(self, random_forest_result: pd.DataFrame, 
                                 main_folder: str, dataset_folder: str) -> None:
        """
        Saves a bar plot of random forest feature importances.

        Args:
            random_forest_result (pd.DataFrame): The random forest results to plot.
            main_folder (str): The main folder where the plot will be saved.
            dataset_folder (str): The subfolder within the main folder to save the plot.
        """
        path_to_folder = os.path.join(main_folder, dataset_folder)
        os.makedirs(path_to_folder, exist_ok=True)
        full_path = os.path.join(path_to_folder, "random_forest_feature_selection.png")

        # Plot random forest scores
        plt.figure(figsize=(24, 8))
        plt.barh(random_forest_result['Feature'], random_forest_result['Importance'], color='skyblue')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importances from Random Forest')
        plt.gca().invert_yaxis()

        plt.tight_layout()
        plt.savefig(full_path, dpi = self.__dpi)
        plt.close()



    def __plot_clusters_for_column(self, dataset: pd.DataFrame, column: str, 
                                    clusters: np.ndarray, filename: str) -> None:
        """
        Generates count plots for the specified column, visualizing the distribution of 
        responses for each cluster in the dataset. It saves the resulting plots to the specified filename.

        Args:
            dataset (pd.DataFrame): The dataset containing the data to plot.
            column (str): The column for which clusters will be plotted.
            clusters (np.ndarray): Array of cluster labels representing different clusters in the dataset.
            filename (str): Filename for saving the plot, including the file extension (e.g., '.png').

        """
        n_cols = self.__n_columns
        n_rows = self.__n_rows
        n_plots = n_cols * n_rows

        _, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10))
        axes = axes.flatten()

        for i, cluster in enumerate(clusters):
            cluster_data = dataset[dataset['Cluster'] == cluster]
            top_categories = cluster_data[column].value_counts().index[:20]

            sns.countplot(y=cluster_data[column], ax=axes[i], order=top_categories, palette='magma')
            axes[i].set_title(f"Cluster {cluster + 1}")
            axes[i].set_xlabel('Count')
            axes[i].set_ylabel('Answers')
            axes[i].text(0.95, 0.05, column, horizontalalignment='right', verticalalignment='bottom', 
                        transform=axes[i].transAxes, fontsize=8, color='black', style='normal')

        # Turn off empty subplots if clusters are over
        for j in range(len(clusters), n_plots):
            axes[j].axis('off')

        plt.tight_layout()
        plt.savefig(filename, dpi = self.__dpi)
        plt.close()

    def plot_important_features_table(self, top_n_ranked_features: pd.DataFrame, 
                                  main_folder: str, dataset_folder: str) -> None:
        """
        Creates a table visualization of the most important features, ranked by their importance. 

        Args:
            top_n_ranked_features (pd.DataFrame): DataFrame containing feature names and their ranks, 
            where lower ranks signify higher importance.
            main_folder (str): The main folder where the table image will be saved.
            dataset_folder (str): The subfolder within the main folder to save the table image.
        """
        path_to_folder = os.path.join(main_folder, dataset_folder)
        os.makedirs(path_to_folder, exist_ok=True)
        full_path = os.path.join(path_to_folder, "table_image.png")

        top_n_ranked_features = top_n_ranked_features.reset_index()
        top_n_ranked_features = top_n_ranked_features.rename(columns={'index': 'Feature'})

        _, ax = plt.subplots(figsize=(8, 4))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=top_n_ranked_features.values, colLabels=top_n_ranked_features.columns, 
                         cellLoc='center', loc='center')
        for i in range(len(top_n_ranked_features)):
            table[(i + 1, 0)].set_text_props(ha='left')
        table.auto_set_column_width(range(len(top_n_ranked_features.columns)))
        table.auto_set_font_size(False)
        table.set_fontsize(7)        
        table.scale(2, 1.5)

        plt.tight_layout()
        plt.savefig(full_path, bbox_inches='tight', dpi = self.__dpi)
        plt.close()
    

    def plot_important_features(self, dataset: pd.DataFrame, columns_to_plot: list[str], 
                                main_folder: str, dataset_folder: str, test_name: str) -> None:
        """
        Plots important features for each cluster in the dataset and saves the resulting plots.

        Args:
            dataset (pd.DataFrame): The dataset containing clusters and features to plot.
            columns_to_plot (list[str]): List of dataset column names to plot.
            main_folder (str): The main folder where the plots will be saved.
            dataset_folder (str): The subfolder within the main folder to save the plots.
            test_name (str): Name for the test used in the plots' folder structure.
        """
        path_to_folder = os.path.join(main_folder, dataset_folder, test_name)
        os.makedirs(path_to_folder, exist_ok=True)

        clusters = dataset['Cluster'].unique()
        clusters.sort()
        columns_to_plot = dataset[columns_to_plot]
        i=1
        for column in columns_to_plot:
            column_dir = os.path.join(path_to_folder, f"top_feature_{i}")
            os.makedirs(column_dir, exist_ok=True)

            batch_size = 4
            for start_idx in range(0, len(clusters), batch_size):
                end_idx = min(start_idx + batch_size, len(clusters))
                clusters_to_plot = clusters[start_idx:end_idx]

                filename = os.path.join(column_dir, f"{i}_feature_clusters_{start_idx + 1}_to_{end_idx}.png")
                self.__plot_clusters_for_column(dataset, column, clusters_to_plot, filename)
                plt.close()
            i+=1
