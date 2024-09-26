from WorkingDatasetInfo import WorkingDatasetInfo
from DatasetAnalyzer import DatasetAnalyzer
from TextFeatureExtractor import TextFeatureExtractor
from GraphPlotter import GraphPlotter
from DataEncoder import DataEncoder
from OptimalClusterFinder import OptimalClusterFinder
from DimensionalityReducer import DimensionalityReducer
from DataClusterer import DataClusterer
from ResultInterpreter import ResultInterpreter

import pandas as pd
import numpy as np
import json



with open('config.json', 'r') as f:
    config = json.load(f)

# For preprocessing data
path_to_dataset = config['WorkingDataset']['path_to_dataset']
missing_information_max_percent = config['WorkingDataset']['missing_information_max_percent']

# Settings for depiction results (form, the number of features and the quality of an image)
n_columns = config['GraphPlotter']['n_columns']
n_rows = config['GraphPlotter']['n_rows']
max_feature_number = config['GraphPlotter']['max_feature_number_to_plot']
graph_dpi = config['GraphPlotter']['dpi']

# Paths to save all results before interpreting
path_to_original_graphs = config['GraphPlotter']['path_to_original_graphs']
path_to_processed_graphs = config['GraphPlotter']['path_to_processed_graphs']
path_to_cluster_choice_graphs = config['GraphPlotter']['path_to_cluster_choice_graphs']
path_to_k_elbow = config['GraphPlotter']['path_to_k_elbow']
path_to_silhouette_score = config['GraphPlotter']['path_to_silhouette_score']
path_to_dendrogram = config['GraphPlotter']['path_to_dendrogram']
path_to_bic_aic = config['GraphPlotter']['path_to_bic_aic']

path_to_reduced_components_visualization = config['GraphPlotter']['path_to_reduced_components_visualization']
path_to_cluster_results = config['GraphPlotter']['path_to_cluster_results']

# Setting determines whether text columns are considered during encoding and clustering 
with_text_columns = bool(config['WorkingDataset']['with_text_columns'])

# Settings for dimensionality reduction
save_info_ratio = config["DimensionalityReducer"]["save_info_ratio"]
tsne_random_state = config["DimensionalityReducer"]["tsne_random_state"]

# Settings for clustering
cluster_number = config["Clusterization"]["cluster_number"]
kmeans_init = config["Clusterization"]["kmeans_init"]
kmeans_random_state = config["Clusterization"]["kmeans_random_state"]
gauss_random_state = config["Clusterization"]["gauss_random_state"]

# Settings for tuning permutation feature importance for random forest algorithm
random_forest_with_permutations = bool(config["ResultInterpreter"]["random_forest_with_permutations"])
random_forest_permutation_repeats = config["ResultInterpreter"]["random_forest_permutation_repeats"]
permutation_random_state = config["ResultInterpreter"]["permutation_random_state"]

# Paths to save cluster interpreting results
path_to_interpretations = config['GraphPlotter']['path_to_interpretations']
path_to_pca_gauss_result = config["ResultInterpreter"]["path_to_pca_gauss_result"]
path_to_tsne_gauss_result = config["ResultInterpreter"]["path_to_tsne_gauss_result"]
path_to_pca_kmeans_result = config["ResultInterpreter"]["path_to_pca_kmeans_result"]
path_to_tsne_kmeans_result = config["ResultInterpreter"]["path_to_tsne_kmeans_result"]
path_to_pca_agglomerative_result = config["ResultInterpreter"]["path_to_pca_agglomerative_result"]
path_to_tsne_agglomerative_result = config["ResultInterpreter"]["path_to_tsne_agglomerative_result"]



if __name__ == "__main__":
    original_dataset = pd.read_csv(path_to_dataset)
    dataset_info = WorkingDatasetInfo(original_dataset)
    analyzer = DatasetAnalyzer(original_dataset)
    plotter = GraphPlotter(n_columns, n_rows, max_feature_number, graph_dpi)
    cluster_finder = OptimalClusterFinder(kmeans_init, kmeans_random_state, graph_dpi)
    dimension_reducer = DimensionalityReducer(tsne_random_state)

    # Getting dataset basic info
    dataset_info.print_dataset_info()
    # dataset_info.print_each_column_types()
    analyzer.check_missing_values(percent_threshold=missing_information_max_percent)
    plotter.save_plots(path_to_original_graphs, original_dataset)
    print("Analysis complete!")

    # Analyzing and preparing data for future working
    analyzer.drop_sparse_columns()
    analyzer.preprocess_columns()
    categorical_dataset, text_dataset = analyzer.return_divided_datasets()
    feature_extractor = TextFeatureExtractor(text_dataset)
    feature_extractor.extract_features()
    print("Data preparation complete!")

    preprocessed_dataset = pd.concat([categorical_dataset, text_dataset], axis = 1)
    plotter.save_plots(path_to_processed_graphs, preprocessed_dataset)
    print("Graphs with prepared data saved!")

    # Encoding data for machine learning algorithms to work
    encoder = DataEncoder()
    encoder.pass_text_columns(text_dataset.columns)
    encoder.encode_data(preprocessed_dataset, with_text_columns)
    encoder.normalize_data()
    encoded_dataset, normalized_dataset = encoder.get_encoded_dataset()
    print("Data encoded!")



    # Finding the optimal number of clusters
    def find_optimal_k(dataset: pd.DataFrame, type_of_dataset: str, dendrogram_threshold: int) -> None:
        """
        Finds the optimal number of clusters for the given dataset using various methods and saves the corresponding plots.

        This function employs the following methods to determine the optimal number of clusters:
            1. Elbow method
            2. Silhouette score method
            3. Dendrogram method
            4. BIC/AIC for Gaussian Mixture Model

        Args:
            dataset (pd.DataFrame): The dataset to be clustered, represented as a pandas DataFrame.
            type_of_dataset (str): A string indicating the type of dataset (e.g., 'encoded', 'normalized').
            dendrogram_threshold (int): The threshold to be used in the dendrogram method for cluster separation.
        """
        cluster_finder.k_elbow_method(dataset=dataset, 
                                    folder_path=path_to_cluster_choice_graphs,
                                    k_elbow_path = path_to_k_elbow,
                                    file_name=f"{type_of_dataset}_k_elbow_method")
        cluster_finder.silhouette_score_method(dataset = dataset,
                                            folder_path = path_to_cluster_choice_graphs,
                                            silhouette_score_path = f"{path_to_silhouette_score}/on_{type_of_dataset}_dataset",
                                            file_name = f"{type_of_dataset}_silhouette")
        cluster_finder.dendrogram_method(dataset = dataset,
                                            folder_path = path_to_cluster_choice_graphs,
                                            dendrogram_path = path_to_dendrogram,
                                            file_name = f"{type_of_dataset}_dendrogram", threshold = dendrogram_threshold)
        cluster_finder.bic_aic_for_gaussian_method(dataset = dataset,
                                                    folder_path = path_to_cluster_choice_graphs,
                                                    bic_aic_path = path_to_bic_aic,
                                                    file_name = f"{type_of_dataset}_bic_aic",
                                                    gauss_random_state = gauss_random_state)

    # Reducing dimensionality to save n% of information
    pca_95_result = dimension_reducer.get_n_percent_pca_result(saved_info_ratio = save_info_ratio, normalized_dataset = normalized_dataset)

    find_optimal_k(dataset = encoded_dataset.values, type_of_dataset = "encoded", dendrogram_threshold = 60)
    find_optimal_k(dataset = normalized_dataset, type_of_dataset = "normalized", dendrogram_threshold = 62)
    find_optimal_k(dataset = pca_95_result, type_of_dataset = "pca", dendrogram_threshold = 65)
    print("Cluster number evaluation complete!\n")



    # Reducing dimensionality to 2D and 3D for clusterization
    # Linear PCA
    norm_pca_2d_result, norm_pca_3d_result = dimension_reducer.\
        get_linear_pca_result(normalized_dataset = normalized_dataset)
    # Kernel PCA and t-SNE for normalized data
    norm_tsne_2d_result, norm_tsne_3d_result = dimension_reducer.\
        get_tsne_result(normalized_or_pca_dataset = normalized_dataset)
    norm_kernel_pca_2d_result, norm_kernel_pca_3d_result = dimension_reducer.\
        get_kernel_pca_result(normalized_or_pca_dataset = normalized_dataset)
    # Kernel PCA and t-SNE for data reduced by linear PCA with 95% information saved
    pca_tsne_2d_result, pca_tsne_3d_result = dimension_reducer.\
        get_tsne_result(normalized_or_pca_dataset = pca_95_result)
    pca_kernel_pca_2d_result, pca_kernel_pca_3d_result = dimension_reducer.\
        get_kernel_pca_result(normalized_or_pca_dataset = pca_95_result)

    # Save reduced data visualizations
    # Linear pca in 2d and 3d
    plotter.save_2d_reduced_data_plotes(path_to_save = path_to_reduced_components_visualization, file_name = "linear_pca_2d.png",
                                        reducing_method="Linear_PCA_2D", reduced_data = norm_pca_2d_result)
    plotter.save_3d_reduced_data_plotes(path_to_save = path_to_reduced_components_visualization, file_name = "linear_pca_3d.png",
                                        reducing_method="Linear_PCA_3D", reduced_data = norm_pca_3d_result)
    # Kernel pca in 2d and 3d on normalized and pca-reduced data
    plotter.save_2d_reduced_data_plotes(path_to_save = path_to_reduced_components_visualization, file_name = "norm_kernel_pca_2d.png",
                                        reducing_method="Kernel_PCA_2D", reduced_data = norm_kernel_pca_2d_result)
    plotter.save_3d_reduced_data_plotes(path_to_save = path_to_reduced_components_visualization, file_name = "norm_kernel_pca_3d.png",
                                        reducing_method="Kernel_PCA_3D", reduced_data = norm_kernel_pca_3d_result)
    plotter.save_2d_reduced_data_plotes(path_to_save = path_to_reduced_components_visualization, file_name = "pca_kernel_pca_2d.png",
                                        reducing_method="Kernel_PCA_2D", reduced_data = pca_kernel_pca_2d_result)
    plotter.save_3d_reduced_data_plotes(path_to_save = path_to_reduced_components_visualization, file_name = "pca_kernel_pca_3d.png",
                                        reducing_method="Kernel_PCA_3D", reduced_data = pca_kernel_pca_3d_result)
    # t-SNE in 2d and 3d on normalized and pca-reduced data
    plotter.save_2d_reduced_data_plotes(path_to_save = path_to_reduced_components_visualization, file_name = "norm_tsne_2d.png",
                                        reducing_method="tSNE_2D", reduced_data = norm_tsne_2d_result)
    plotter.save_3d_reduced_data_plotes(path_to_save = path_to_reduced_components_visualization, file_name = "norm_tsne_3d.png",
                                        reducing_method="tSNE_3D", reduced_data = norm_tsne_3d_result)
    plotter.save_2d_reduced_data_plotes(path_to_save = path_to_reduced_components_visualization, file_name = "pca_tsne_2d.png",
                                        reducing_method="tSNE_2D", reduced_data = pca_tsne_2d_result)
    plotter.save_3d_reduced_data_plotes(path_to_save = path_to_reduced_components_visualization, file_name = "pca_tsne_3d.png",
                                        reducing_method="tSNE_3D", reduced_data = pca_tsne_3d_result)
    # Showing the aproximate form of a t-SNE group 
    component_range = (10, 20)
    plotter.save_3d_reduced_data_plot_with_range(path_to_save = path_to_reduced_components_visualization, 
                                                 file_name = "norm_tsne_3d_range_0_90.png", component_range = component_range,
                                                 reducing_method="tSNE_3D", reduced_data = norm_tsne_3d_result, elev = 0, azim = 90)
    plotter.save_3d_reduced_data_plot_with_range(path_to_save = path_to_reduced_components_visualization, 
                                                 file_name = "norm_tsne_3d_range_0_45.png", component_range = component_range,
                                                 reducing_method="tSNE_3D", reduced_data = norm_tsne_3d_result, elev = 0, azim = 45)
    plotter.save_3d_reduced_data_plot_with_range(path_to_save = path_to_reduced_components_visualization, 
                                                 file_name = "norm_tsne_3d_range_0_0.png", component_range = component_range,
                                                 reducing_method="tSNE_3D", reduced_data = norm_tsne_3d_result, elev = 0, azim = 0)
    print("Dimensionality reduction complete!\n")



    # Clusterization
    clusterer = DataClusterer(cluster_number = cluster_number,
                              kmeans_random_state = kmeans_random_state,
                              kmeans_init = kmeans_init,
                              gauss_random_state = gauss_random_state)

    # dataset_to_copy = dataset
    if not with_text_columns:
        dataset_to_copy = preprocessed_dataset.drop(columns = text_dataset.columns)
    else: dataset_to_copy = preprocessed_dataset
    def classify_dataset(reduced_dataset: np.ndarray, 
                        method_of_clustering: callable) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Classifies a given reduced dataset using a specified clustering method.

        This function applies a clustering method to the reduced dataset and appends the resulting cluster labels 
        to a copy of the original dataset.

        Args:
            reduced_dataset (np.ndarray): The input dataset that has been reduced, represented as a NumPy array.
            method_of_clustering (callable): A callable that takes the reduced dataset as input and returns cluster labels.

        Returns:
            tuple[pd.DataFrame, np.ndarray]: A tuple containing:
                - result_dataset (pd.DataFrame): A copy of the original dataset with an additional column 'Cluster' 
                containing the cluster labels.
                - cluster_labels (np.ndarray): An array of cluster labels assigned to each data point in the reduced dataset.
        """
        cluster_labels = method_of_clustering(reduced_dataset)
        result_dataset = dataset_to_copy.copy()
        result_dataset['Cluster'] = cluster_labels
        return result_dataset, cluster_labels

    # Gaussian Mixture Model
    pca_gaussian_dataset, pca_gaussian_labels = classify_dataset(reduced_dataset = norm_pca_2d_result,
                                                                 method_of_clustering = clusterer.gaussian_mixture_clusterization)
    tsne_gaussian_dataset, tsne_gaussian_labels = classify_dataset(reduced_dataset = norm_tsne_2d_result,
                                                                 method_of_clustering = clusterer.gaussian_mixture_clusterization)
    # K-Means
    pca_kmeans_dataset, pca_kmeans_cluster_labels = classify_dataset(reduced_dataset = pca_tsne_2d_result,
                                                                     method_of_clustering = clusterer.kmeans_clusterization)
    tsne_kmeans_dataset, tsne_kmeans_cluster_labels = classify_dataset(reduced_dataset = norm_tsne_2d_result,
                                                                     method_of_clustering = clusterer.kmeans_clusterization)
    # Agglomerative clustering
    pca_agglomerative_dataset, pca_agglomerative_cluster_labels = classify_dataset(reduced_dataset = pca_tsne_2d_result,
                                                                     method_of_clustering = clusterer.agglomerative_clusterization)
    tsne_agglomerative_dataset, tsne_agglomerative_cluster_labels = classify_dataset(reduced_dataset = norm_tsne_2d_result,
                                                                     method_of_clustering = clusterer.agglomerative_clusterization)

    # Save clustering results
    # Gaussian Mixture Model
    plotter.save_clustering_plots(path_to_save = path_to_cluster_results, file_name = "gaussian_on_norm_pca_2d.png", 
                                  type_of_clustering = "Gaussian", reducing_method="PCA", 
                                  reduced_data = norm_pca_2d_result, cluster_labels = pca_gaussian_labels)
    plotter.save_clustering_plots(path_to_save = path_to_cluster_results, file_name = "gaussian_on_norm_tsne_2d.png", 
                                  type_of_clustering = "Gaussian", reducing_method="t-SNE", 
                                  reduced_data = norm_tsne_2d_result, cluster_labels = tsne_gaussian_labels)
    # K-Means
    plotter.save_clustering_plots(path_to_save = path_to_cluster_results, file_name = "kmeans_on_pca_tsne_2d.png", 
                                  type_of_clustering = "KMeans", reducing_method="t-SNE", 
                                  reduced_data = pca_tsne_2d_result, cluster_labels = pca_kmeans_cluster_labels)
    plotter.save_clustering_plots(path_to_save = path_to_cluster_results, file_name = "kmeans_on_norm_tsne_2d.png", 
                                  type_of_clustering = "KMeans", reducing_method="t-SNE", 
                                  reduced_data = norm_tsne_2d_result, cluster_labels = tsne_kmeans_cluster_labels)
    # Agglomerative clustering
    plotter.save_clustering_plots(path_to_save = path_to_cluster_results, file_name = "agglomerative_on_pca_tsne_2d.png", 
                                  type_of_clustering = "Agglomerative", reducing_method="t-SNE", 
                                  reduced_data = pca_tsne_2d_result, cluster_labels = pca_agglomerative_cluster_labels)
    plotter.save_clustering_plots(path_to_save = path_to_cluster_results, file_name = "agglomerative_on_norm_tsne_2d.png", 
                                  type_of_clustering = "Agglomerative", reducing_method="t-SNE", 
                                  reduced_data = norm_tsne_2d_result, cluster_labels = tsne_agglomerative_cluster_labels)    
    print("Clusterization complete!\n")



    # Interpreting
    def interpret_clusterisation(dataset_to_interpret: pd.DataFrame, result_dataset_name: str, dataset_folder: str):
        interpreter = ResultInterpreter(dataset = dataset_to_interpret, result_dataset_name = result_dataset_name,
                                             main_folder = path_to_interpretations, dataset_folder = dataset_folder)
        #Correlation matrix
        correlation_matrix = interpreter.get_correlation_matrix()
        plotter.plot_correlation_matrix(correlation_matrix = correlation_matrix, main_folder = path_to_interpretations, 
                                      dataset_folder = dataset_folder)
        #Chi2
        chi2_result = interpreter.chi_squared_feature_selection()
        plotter.save_chi_squared_plot(chi2_result = chi2_result[:max_feature_number], main_folder = path_to_interpretations, 
                                      dataset_folder = dataset_folder)
        plotter.plot_important_features(dataset = dataset_to_interpret, columns_to_plot = chi2_result["Feature"][:max_feature_number],
                                        main_folder = path_to_interpretations, dataset_folder = dataset_folder,
                                        test_name = "chi_squared_selected_features")
        #Mutual information
        mutual_info_result = interpreter.mutual_info_feature_selection()
        plotter.save_mutual_info_plot(mutual_info_result = mutual_info_result[:max_feature_number], 
                                      main_folder = path_to_interpretations, dataset_folder = dataset_folder)
        plotter.plot_important_features(dataset = dataset_to_interpret, columns_to_plot = mutual_info_result["Feature"][:max_feature_number],
                                        main_folder = path_to_interpretations, dataset_folder = dataset_folder,
                                        test_name = "mutual_information_selected_features")
        #Random Forest
        random_forest_result = interpreter.random_forest_feature_selection(with_permutations = random_forest_with_permutations,
                                                                           permutation_repeats = random_forest_permutation_repeats,
                                                                           permutation_random_state = permutation_random_state)
        plotter.save_random_forest_plot(random_forest_result = random_forest_result[:max_feature_number], 
                                        main_folder = path_to_interpretations, dataset_folder = dataset_folder)
        plotter.plot_important_features(dataset = dataset_to_interpret, columns_to_plot = random_forest_result["Feature"][:max_feature_number],
                                        main_folder = path_to_interpretations, dataset_folder = dataset_folder,
                                        test_name = "random_forest_selected_features")

    interpret_clusterisation(dataset_to_interpret = pca_gaussian_dataset, result_dataset_name = "pca_gaussian_dataset.csv",
                             dataset_folder = path_to_pca_gauss_result)
    interpret_clusterisation(dataset_to_interpret = tsne_gaussian_dataset, result_dataset_name = "tsne_gaussian_dataset.csv",
                             dataset_folder = path_to_tsne_gauss_result)

    interpret_clusterisation(dataset_to_interpret = pca_kmeans_dataset, result_dataset_name = "pca_kmeans_dataset.csv",
                             dataset_folder = path_to_pca_kmeans_result)
    interpret_clusterisation(dataset_to_interpret = tsne_kmeans_dataset, result_dataset_name = "tsne_kmeans_dataset.csv",
                             dataset_folder = path_to_tsne_kmeans_result)

    interpret_clusterisation(dataset_to_interpret = pca_agglomerative_dataset, result_dataset_name = "pca_agglomerative_dataset.csv",
                             dataset_folder = path_to_pca_agglomerative_result)
    interpret_clusterisation(dataset_to_interpret = tsne_agglomerative_dataset, result_dataset_name = "tsne_agglomerative_dataset.csv",
                             dataset_folder = path_to_tsne_agglomerative_result)    
    print("Interpretation complete!") 
