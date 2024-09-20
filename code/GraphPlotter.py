import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

class GraphPlotter:
    def __init__(self, n_columns: int, n_rows: int, max_feature_number: int):
        self.__n_columns = n_columns
        self.__n_rows = n_rows
        self.__max_feature_number = max_feature_number

    def __display_data_per_question(self, dataset: pd.DataFrame, column_index: int, ax):
        column = dataset.columns[column_index]  # Получаем название столбца по индексу
        top_20_answers = dataset[column].value_counts().nlargest(self.__max_feature_number).index
        # Создаем график только для первых 20 значений
        sns.countplot(y=column, data=dataset[dataset[column].isin(top_20_answers)], ax=ax, order=top_20_answers, palette='magma')
    
        # sns.countplot(y=column, data=dataset, ax=ax, order=dataset[column].value_counts().index, palette='magma')
        ax.set_xlabel('Count', fontsize=9)
        ax.set_ylabel('Answers', fontsize=9)
        ax.tick_params(axis='x', labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
        ax.text(0.95, 0.05, column, horizontalalignment='right', verticalalignment='bottom', 
            transform=ax.transAxes, fontsize=8, color='black', style='normal')

    def __plot_columns_in_batches(self, dataset: pd.DataFrame, start_idx: int, end_idx: int, filename: str, function):
        n_columns = self.__n_columns  # Number of columns in the grid
        n_rows = self.__n_rows  # Number of rows in the grid
        n_plots = n_columns * n_rows

        # Create a new figure for each batch
        fig, axes = plt.subplots(n_rows, n_columns, figsize=(30, 10))  # Adjust size as needed
        axes = axes.flatten()  # Flatten the axes array for easy iteration
        for i in range(start_idx, min(end_idx, start_idx + n_plots)):
            function(dataset, i, axes[i - start_idx])
        # Hide unused subplots if the number of plots is less than n_plots
        for j in range(end_idx - start_idx, n_plots):
            axes[j].axis('off')

        plt.tight_layout()
        plt.savefig(filename, dpi=300)  # Save the plot as an image file
        plt.close(fig)  # Close the figure to free up memory

    def save_plots(self, path_to_save: str, dataset: pd.DataFrame) -> None:
        os.makedirs(path_to_save, exist_ok=True)
        dataset = dataset.fillna('NaN')

        relative_path = path_to_save
        total_columns = len(dataset.columns)
        batch_size = self.__n_columns*self.__n_rows
        for start_idx in range(0, total_columns, batch_size):
            end_idx = min(start_idx + batch_size, total_columns)
            filename = f"{relative_path}/{start_idx + 1}_{end_idx}_questions_from_text_data.png"
            self.__plot_columns_in_batches(dataset, start_idx, end_idx, filename, self.__display_data_per_question)
  
    def save_reduced_data_plotes(self, path_to_save: str, file_name: str, 
                                 reducing_method: str, reduced_data: np.ndarray):
        os.makedirs(path_to_save, exist_ok=True)
        full_path = os.path.join(path_to_save, file_name)

        plt.figure(figsize=(10, 8))
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=reduced_data[:, 0], cmap='tab10', s=50, alpha=0.7)
        plt.title(f"{reducing_method} Visualization")
        plt.xlabel(f"{reducing_method} Component 1")
        plt.ylabel(f"{reducing_method} Component 2")
        plt.colorbar(label="Cluster Label")
        plt.savefig(full_path, dpi = 300)
        plt.close()

    def save_clustering_plots(self, path_to_save: str, file_name: str, type_of_clustering: str, 
                              reducing_method: str, reduced_data: np.ndarray, cluster_labels: np.ndarray):
        os.makedirs(path_to_save, exist_ok=True)
        full_path = os.path.join(path_to_save, file_name)

        plt.figure(figsize=(10, 8))
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels, cmap='tab10', s=50, alpha=0.7)
        plt.title(f"{type_of_clustering} Visualization")
        plt.xlabel(f"{reducing_method} Component 1")
        plt.ylabel(f"{reducing_method} Component 2")
        plt.colorbar(label="Cluster Label")
        plt.savefig(full_path, dpi = 300)
        plt.close()

    

    def plot_correlation_matrix(self, correlation_matrix: pd.DataFrame, main_folder: str, dataset_folder: str):
        path_to_folder = os.path.join(main_folder, dataset_folder)
        os.makedirs(path_to_folder, exist_ok=True)
        full_path = os.path.join(path_to_folder, "correlation_matrix.png")

        plt.figure(figsize=(24, 24))
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Matrix')
        # plt.xticks(rotation=-15)
        # plt.yticks(rotation=-15)

        plt.tight_layout()
        plt.savefig(full_path, dpi = 300)

    def save_chi_squared_plot(self, chi2_result: pd.DataFrame, main_folder: str, dataset_folder: str):
        path_to_folder = os.path.join(main_folder, dataset_folder)
        os.makedirs(path_to_folder, exist_ok=True)
        full_path = os.path.join(path_to_folder, "chi_squared__feature selection.png")

        # Plot Chi-Squared values
        plt.figure(figsize=(24, 8))
        sns.barplot(x='Chi-Squared', y='Feature', data=chi2_result.head(20), palette='viridis')
        plt.title('Top 20 Features by Chi-Squared Statistic')
        plt.xlabel('Chi-Squared Value')
        plt.ylabel('Feature')
        
        plt.tight_layout()
        plt.savefig(full_path, dpi = 300)

    def save_mutual_info_plot(self, mutual_info_result: pd.DataFrame, main_folder: str, dataset_folder: str):
        path_to_folder = os.path.join(main_folder, dataset_folder)
        os.makedirs(path_to_folder, exist_ok=True)
        full_path = os.path.join(path_to_folder, "mutual_info_feature selection.png")
    
        # Plot mutual information scores
        plt.figure(figsize=(24, 8))
        plt.barh(mutual_info_result['Feature'], mutual_info_result['Score'], color='skyblue')
        plt.title('Mutual Information Scores of Features')
        plt.xlabel('Mutual Information Score')
        plt.ylabel('Features')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(full_path, dpi = 300)

    def save_random_forest_plot(self, random_forest_result: pd.DataFrame, main_folder: str, dataset_folder: str):
        path_to_folder = os.path.join(main_folder, dataset_folder)
        os.makedirs(path_to_folder, exist_ok=True)
        full_path = os.path.join(path_to_folder, "random_forest_feature selection.png")

        # Plot random forest scores
        plt.figure(figsize=(24, 8))
        plt.barh(random_forest_result['Feature'], random_forest_result['Importance'], color='skyblue')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importances from Random Forest')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(full_path, dpi = 300)

    # def __plot_clusters_for_column(self, dataset: pd.DataFrame, column: str, clusters, filename: str):
    #     n_cols = self.__n_columns  # Количество столбцов на графике
    #     n_rows = self.__n_rows  # Количество строк на графике
    #     n_plots = n_cols * n_rows  # Всего 4 графика на картинке

    #     fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10))
    #     axes = axes.flatten()

    #     for i, cluster in enumerate(clusters):
    #         cluster_data = dataset[dataset['Cluster'] == cluster]
    #         top_categories = cluster_data[column].value_counts().index[:20]

    #         sns.countplot(y=cluster_data[column], ax=axes[i], order=top_categories, palette='magma')
    #         axes[i].set_title(f"Cluster {cluster}")
    #         axes[i].set_xlabel('Count')
    #         axes[i].set_ylabel('Answers')
    #         axes[i].text(0.95, 0.05, column, horizontalalignment='right', verticalalignment='bottom', 
    #                     transform=axes[i].transAxes, fontsize=8, color='black', style='normal')

    #         # Отключаем пустые субплоты, если кластеры закончились
    #     for j in range(len(clusters), n_plots):
    #         axes[j].axis('off')

    #     plt.tight_layout()
    #     plt.savefig(filename, dpi=300)
    #     plt.close()

    # def plot_important_features(self, path_to_folder: str, columns_to_plot):
    #     clusters = result_dataset['Cluster'].unique()
    #     clusters.sort()
    #     output_dir = path_to_folder
    #     os.makedirs(output_dir, exist_ok=True)

    #     columns_to_plot = result_dataset[columns_to_plot]
    #     i=1
    #     for column in columns_to_plot:
    #         column_dir = os.path.join(output_dir, f"question_{i}")
    #         os.makedirs(column_dir, exist_ok=True)

    #         # Разбиваем кластеры на группы по 4 для построения на одной картинке
    #         batch_size = 4
    #         for start_idx in range(0, len(clusters), batch_size):
    #             end_idx = min(start_idx + batch_size, len(clusters))
    #             clusters_to_plot = clusters[start_idx:end_idx]

    #             # Имя файла для каждого столбца и группы кластеров
    #             filename = os.path.join(column_dir, f"{i}_question_clusters_{start_idx + 1}_to_{end_idx}.png")

    #             # Строим графики для текущего столбца по выбранным кластерам
    #             self.__plot_clusters_for_column(result_dataset, column, clusters_to_plot, filename)
    #         i+=1