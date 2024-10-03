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
path_to_dataset = config['DataPreprocessingParameters']['path_to_dataset']
missing_information_max_percent = config['DataPreprocessingParameters']['missing_information_max_percent']

# Settings for depiction results (form, the number of features and the quality of an image)
n_columns = config['GraphPlotterGridParameters']['n_columns']
n_rows = config['GraphPlotterGridParameters']['n_rows']
max_feature_number = config['GraphPlotterGridParameters']['max_feature_number_to_plot']
graph_dpi = config['GraphPlotterGridParameters']['dpi']

# Paths to save all results before interpreting
path_to_original_graphs = config['GraphPlotterSavePaths']['path_to_original_graphs']
path_to_processed_graphs = config['GraphPlotterSavePaths']['path_to_processed_graphs']
path_to_cluster_choice_graphs = config['GraphPlotterSavePaths']['path_to_cluster_choice_graphs']
path_to_k_elbow = config['GraphPlotterSavePaths']['path_to_k_elbow']
path_to_silhouette_score = config['GraphPlotterSavePaths']['path_to_silhouette_score']
path_to_dendrogram = config['GraphPlotterSavePaths']['path_to_dendrogram']
path_to_bic_aic = config['GraphPlotterSavePaths']['path_to_bic_aic']

path_to_reduced_components_visualization = config['GraphPlotterSavePaths']['path_to_reduced_components_visualization']
path_to_cluster_results = config['GraphPlotterSavePaths']['path_to_cluster_results']

# Setting determines whether text columns are considered during encoding and clustering 
with_text_columns = bool(config['AdditionalParamters']['with_text_columns'])

# Settings for dimensionality reduction
save_info_ratio = config["DimensionalityReducerParameters"]["save_info_ratio"]
pca_random_state = config["DimensionalityReducerParameters"]["pca_random_state"]
kernel_pca_random_state = config["DimensionalityReducerParameters"]["kernel_pca_random_state"]
tsne_random_state = config["DimensionalityReducerParameters"]["tsne_random_state"]
mds_random_state = config["DimensionalityReducerParameters"]["mds_random_state"]

tsne_slice_range = tuple(config["DimensionalityReducerParameters"]["tsne_slice_range"])
linear_pca_slice_range = tuple(config["DimensionalityReducerParameters"]["linear_pca_slice_range"])
kernel_pca_slice_range = tuple(config["DimensionalityReducerParameters"]["kernel_pca_slice_range"])
mds_slice_range = tuple(config["DimensionalityReducerParameters"]["mds_slice_range"])

# Settings for clustering
cluster_number = config["ClusteringParameters"]["cluster_number"]
kmeans_init = config["ClusteringParameters"]["kmeans_init"]
kmeans_random_state = config["ClusteringParameters"]["kmeans_random_state"]
gauss_random_state = config["ClusteringParameters"]["gauss_random_state"]

# Settings for tuning feature selection process
random_forest_with_permutations = bool(config["ResultInterpreterParameters"]["random_forest_with_permutations"])
random_forest_permutation_repeats = config["ResultInterpreterParameters"]["random_forest_permutation_repeats"]
permutation_random_state = config["ResultInterpreterParameters"]["permutation_random_state"]
mutual_information_random_state = config["ResultInterpreterParameters"]["mutual_information_random_state"]
most_important_features_max_number = config["ResultInterpreterParameters"]["most_important_features_max_number"]
interpret_tsne_reduced_data = bool(config["ResultInterpreterParameters"]["interpret_tsne_reduced_data"])
interpret_pca_reduced_data = bool(config["ResultInterpreterParameters"]["interpret_pca_reduced_data"])
interpret_kernel_pca_reduced_data = bool(config["ResultInterpreterParameters"]["interpret_kernel_pca_reduced_data"])

# Paths to save cluster interpreting results
path_to_interpretations = config['GraphPlotterSavePaths']['path_to_interpretations']

path_to_pca_2d_gauss_result = config["ResultInterpreterSavePaths"]["path_to_pca_2d_gauss_result"]
path_to_kernel_pca_2d_gauss_result = config["ResultInterpreterSavePaths"]["path_to_kernel_pca_2d_gauss_result"]
path_to_tsne_2d_gauss_result = config["ResultInterpreterSavePaths"]["path_to_tsne_2d_gauss_result"]

path_to_pca_2d_kmeans_result = config["ResultInterpreterSavePaths"]["path_to_pca_2d_kmeans_result"]
path_to_kernel_pca_2d_kmeans_result = config["ResultInterpreterSavePaths"]["path_to_kernel_pca_2d_kmeans_result"]
path_to_tsne_2d_kmeans_result = config["ResultInterpreterSavePaths"]["path_to_tsne_2d_kmeans_result"]

path_to_pca_2d_agglomerative_result = config["ResultInterpreterSavePaths"]["path_to_pca_2d_agglomerative_result"]
path_to_kernel_pca_2d_agglomerative_result = config["ResultInterpreterSavePaths"]["path_to_kernel_pca_2d_agglomerative_result"]
path_to_tsne_2d_agglomerative_result = config["ResultInterpreterSavePaths"]["path_to_tsne_2d_agglomerative_result"]



if __name__ == "__main__":
    original_dataset = pd.read_csv(path_to_dataset)
    dataset_info = WorkingDatasetInfo(original_dataset)
    analyzer = DatasetAnalyzer(original_dataset)
    plotter = GraphPlotter(n_columns, n_rows, max_feature_number, graph_dpi)
    cluster_finder = OptimalClusterFinder(kmeans_init, kmeans_random_state, gauss_random_state, graph_dpi)
    dimension_reducer = DimensionalityReducer(pca_random_state = pca_random_state,
                                              kernel_pca_random_state = kernel_pca_random_state,
                                              tsne_random_state = tsne_random_state, 
                                              mds_random_state = mds_random_state)



    # Getting dataset basic info
    dataset_info.print_dataset_info()
    dataset_info.print_each_column_types()
    dataset_info.save_unique_values_with_counts_to_dataset()

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
    preprocessed_dataset.to_csv(".\\preprocessed_dataset.csv", index=False)
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
                                                    file_name = f"{type_of_dataset}_bic_aic")


    # Dimensionality reduction
    # Reducing dimensionality to save n% of information
    pca_n_result = dimension_reducer.get_n_percent_pca_result(saved_info_ratio = save_info_ratio,
                                                              pca_random_state = pca_random_state, 
                                                              normalized_dataset = normalized_dataset)

    find_optimal_k(dataset = encoded_dataset.values, type_of_dataset = "encoded", dendrogram_threshold = 60)
    find_optimal_k(dataset = normalized_dataset, type_of_dataset = "normalized", dendrogram_threshold = 62)
    find_optimal_k(dataset = pca_n_result, type_of_dataset = "pca", dendrogram_threshold = 65)
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
    # MDS for encoded data
    mds_2d_result, mds_3d_result = dimension_reducer.\
        get_mds_result(encoded_dataset = encoded_dataset)
    # Kernel PCA and t-SNE for data reduced by linear PCA with 95% information saved
    pca_tsne_2d_result, pca_tsne_3d_result = dimension_reducer.\
        get_tsne_result(normalized_or_pca_dataset = pca_n_result)
    pca_kernel_pca_2d_result, pca_kernel_pca_3d_result = dimension_reducer.\
        get_kernel_pca_result(normalized_or_pca_dataset = pca_n_result)

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
    # MDS in 2d nad 3d on encoded data
    plotter.save_2d_reduced_data_plotes(path_to_save = path_to_reduced_components_visualization, file_name = "mds_2d.png",
                                        reducing_method="Multidimensional Scaling", reduced_data = mds_2d_result)
    plotter.save_3d_reduced_data_plotes(path_to_save = path_to_reduced_components_visualization, file_name = "mds_3d.png",
                                        reducing_method="Multidimensional Scaling", reduced_data = mds_3d_result)

    # "Slices" of data reduced by various methods
    plotter.save_3d_reduced_data_slice(path_to_save = path_to_reduced_components_visualization, 
                                       file_name = "norm_tsne_3d_range_0_90.png", 
                                       component_range = tsne_slice_range, reducing_method = "tsne_3d", 
                                       reduced_data = norm_tsne_3d_result)
    plotter.save_3d_reduced_data_slice(path_to_save = path_to_reduced_components_visualization, 
                                       file_name = "norm_linear_pca_3d_range_0_90.png", 
                                       component_range = linear_pca_slice_range, reducing_method = "linear_pca_3d", 
                                       reduced_data = norm_pca_3d_result)
    plotter.save_3d_reduced_data_slice(path_to_save = path_to_reduced_components_visualization, 
                                       file_name = "norm_kernel_pca_3d_range_0_90.png", 
                                       component_range = kernel_pca_slice_range, reducing_method = "kernel_pca_3d", 
                                       reduced_data = norm_kernel_pca_3d_result)
    plotter.save_3d_reduced_data_slice(path_to_save = path_to_reduced_components_visualization, 
                                       file_name = "norm_mds_3d_range_0_90.png", 
                                       component_range = mds_slice_range, reducing_method = "mds_3d", 
                                       reduced_data = mds_3d_result)
    print("Dimensionality reduction complete!\n")



    # Clusterization
    clusterer = DataClusterer(cluster_number = cluster_number,
                              kmeans_random_state = kmeans_random_state,
                              kmeans_init = kmeans_init,
                              gauss_random_state = gauss_random_state)

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
    kernel_pca_gaussian_dataset, kernel_pca_gaussian_labels = classify_dataset(reduced_dataset = norm_kernel_pca_2d_result,
                                                                method_of_clustering = clusterer.gaussian_mixture_clusterization)
    # K-Means
    pca_kmeans_dataset, pca_kmeans_cluster_labels = classify_dataset(reduced_dataset = pca_tsne_2d_result,
                                                                method_of_clustering = clusterer.kmeans_clusterization)
    tsne_kmeans_dataset, tsne_kmeans_cluster_labels = classify_dataset(reduced_dataset = norm_tsne_2d_result,
                                                                method_of_clustering = clusterer.kmeans_clusterization)
    kernel_pca_kmeans_dataset, kernel_pca_kmeans_labels = classify_dataset(reduced_dataset = norm_kernel_pca_2d_result,
                                                                method_of_clustering = clusterer.kmeans_clusterization)
    # Agglomerative clustering
    pca_agglomerative_dataset, pca_agglomerative_cluster_labels = classify_dataset(reduced_dataset = pca_tsne_2d_result,
                                                                method_of_clustering = clusterer.agglomerative_clusterization)
    tsne_agglomerative_dataset, tsne_agglomerative_cluster_labels = classify_dataset(reduced_dataset = norm_tsne_2d_result,
                                                                method_of_clustering = clusterer.agglomerative_clusterization)
    kernel_pca_agglomerative_dataset, kernel_pca_agglomerative_labels = classify_dataset(reduced_dataset = norm_kernel_pca_2d_result,
                                                                method_of_clustering = clusterer.agglomerative_clusterization)

    # Save clustering results
    # Gaussian Mixture Model
    plotter.save_clustering_plots(path_to_save = path_to_cluster_results, file_name = "gaussian_on_norm_pca_2d.png", 
                                  type_of_clustering = "Gaussian", reducing_method="PCA", 
                                  reduced_data = norm_pca_2d_result, cluster_labels = pca_gaussian_labels)
    plotter.save_clustering_plots(path_to_save = path_to_cluster_results, file_name = "gaussian_on_norm_kernel_pca_2d.png", 
                                  type_of_clustering = "Gaussian", reducing_method="Kernel PCA", 
                                  reduced_data = norm_kernel_pca_2d_result, cluster_labels = kernel_pca_gaussian_labels)
    plotter.save_clustering_plots(path_to_save = path_to_cluster_results, file_name = "gaussian_on_norm_tsne_2d.png", 
                                  type_of_clustering = "Gaussian", reducing_method="t-SNE", 
                                  reduced_data = norm_tsne_2d_result, cluster_labels = tsne_gaussian_labels)
    # K-Means
    plotter.save_clustering_plots(path_to_save = path_to_cluster_results, file_name = "kmeans_on_pca_tsne_2d.png", 
                                  type_of_clustering = "KMeans", reducing_method="t-SNE", 
                                  reduced_data = pca_tsne_2d_result, cluster_labels = pca_kmeans_cluster_labels)
    plotter.save_clustering_plots(path_to_save = path_to_cluster_results, file_name = "kmeans_on_norm_kernel_pca_2d.png", 
                                  type_of_clustering = "KMeans", reducing_method="Kernel PCA", 
                                  reduced_data = norm_kernel_pca_2d_result, cluster_labels = kernel_pca_kmeans_labels)
    plotter.save_clustering_plots(path_to_save = path_to_cluster_results, file_name = "kmeans_on_norm_tsne_2d.png", 
                                  type_of_clustering = "KMeans", reducing_method="t-SNE", 
                                  reduced_data = norm_tsne_2d_result, cluster_labels = tsne_kmeans_cluster_labels)
    # Agglomerative clustering
    plotter.save_clustering_plots(path_to_save = path_to_cluster_results, file_name = "agglomerative_on_pca_tsne_2d.png", 
                                  type_of_clustering = "Agglomerative", reducing_method="t-SNE", 
                                  reduced_data = pca_tsne_2d_result, cluster_labels = pca_agglomerative_cluster_labels)
    plotter.save_clustering_plots(path_to_save = path_to_cluster_results, file_name = "agglomerative_on_norm_kernel_pca_2d.png", 
                                  type_of_clustering = "Gaussian", reducing_method="Kernel PCA", 
                                  reduced_data = norm_kernel_pca_2d_result, cluster_labels = kernel_pca_agglomerative_labels)
    plotter.save_clustering_plots(path_to_save = path_to_cluster_results, file_name = "agglomerative_on_norm_tsne_2d.png", 
                                  type_of_clustering = "Agglomerative", reducing_method="t-SNE", 
                                  reduced_data = norm_tsne_2d_result, cluster_labels = tsne_agglomerative_cluster_labels)    
    print("Clusterization complete!\n")



    # Interpreting
    def interpret_clusterisation(dataset_to_interpret: pd.DataFrame, result_dataset_name: str, 
                                 dataset_folder: str) -> None:
        """
        Interprets the clustering results by performing feature selection using multiple methods 
        and generating visualizations to aid in the understanding of important features.

        The interpretation includes:
        1. Correlation matrix analysis.
        2. Chi-squared feature selection.
        3. Mutual information feature selection.
        4. Random forest feature selection with optional permutations.
        5. Ranking features based on mean ranks from all feature selection methods.

        Args:
            dataset_to_interpret (pd.DataFrame): The dataset to interpret.
            result_dataset_name (str): The name of the file where the interpreted dataset will be saved.
            dataset_folder (str): The folder path where the interpretation results will be saved.

        Details:
            - Correlation Matrix: Generates and saves a heatmap of feature correlations.
            - Chi-squared Test: Ranks features by their Chi2 score and saves a plot.
            - Mutual Information: Ranks features based on mutual information and saves a plot.
            - Random Forest: Selects important features using a random forest classifier and optionally 
            permutation importance, saving a plot of the results.
            - Ranked Features: Aggregates the results from Chi2, mutual information, and random forest feature 
            selection into a ranked list of top features, saving both a table and detailed plots of these features 
            by cluster.
        """
        interpreter = ResultInterpreter(dataset = dataset_to_interpret, result_dataset_name = result_dataset_name,
                                             main_folder = path_to_interpretations, dataset_folder = dataset_folder)
        #Correlation matrix
        correlation_matrix = interpreter.get_correlation_matrix()
        plotter.plot_correlation_matrix(correlation_matrix = correlation_matrix, main_folder = path_to_interpretations, 
                                      dataset_folder = dataset_folder)

        # Chi_squared (Chi2)
        sorted_chi2_result = interpreter.chi_squared_feature_selection()
        plotter.save_chi_squared_plot(chi2_result = sorted_chi2_result[:max_feature_number], main_folder = path_to_interpretations, 
                                      dataset_folder = dataset_folder)

        # Mutual information
        sorted_mutual_info_result = interpreter.mutual_info_feature_selection(mi_random_state = mutual_information_random_state)
        plotter.save_mutual_info_plot(mutual_info_result = sorted_mutual_info_result[:max_feature_number], 
                                      main_folder = path_to_interpretations, dataset_folder = dataset_folder)

        # Random Forest
        sorted_random_forest_result = interpreter.random_forest_feature_selection(with_permutations = random_forest_with_permutations,
                                                                           permutation_repeats = random_forest_permutation_repeats,
                                                                           permutation_random_state = permutation_random_state)
        plotter.save_random_forest_plot(random_forest_result = sorted_random_forest_result[:max_feature_number], 
                                        main_folder = path_to_interpretations, dataset_folder = dataset_folder)
        
        # Ranking and selecting the most important features from all feature selection methods
        ranked_features = interpreter.get_mean_ranks_for_interpretation_results(sorted_chi2_result = sorted_chi2_result,
                                                                  sorted_mutual_info_result = sorted_mutual_info_result,
                                                                  sorted_random_forest_result = sorted_random_forest_result)
        top_n_ranked_features = ranked_features.sort_values(by='mean_rank')[:most_important_features_max_number]
        # Plot and save a table of the top ranked features
        plotter.plot_important_features_table(top_n_ranked_features = top_n_ranked_features, main_folder = path_to_interpretations,
                                              dataset_folder = dataset_folder)
        # Plot and save detailed visualizations of the top features by cluster
        plotter.plot_important_features(dataset = dataset_to_interpret, columns_to_plot = top_n_ranked_features.index,
                                        main_folder = path_to_interpretations, dataset_folder = dataset_folder,
                                        test_name = f"ranked_top_{most_important_features_max_number}_features")

    # Interpretation for dataset reduced by t-SNE algorithm
    if interpret_tsne_reduced_data:
        # Gaussian clustering
        interpret_clusterisation(dataset_to_interpret = tsne_gaussian_dataset, result_dataset_name = "tsne_gaussian_dataset.csv",
                                dataset_folder = path_to_tsne_2d_gauss_result)
        # K-Means clustering
        interpret_clusterisation(dataset_to_interpret = tsne_kmeans_dataset, result_dataset_name = "tsne_kmeans_dataset.csv",
                                dataset_folder = path_to_tsne_2d_kmeans_result)
        # Agglomerative clustering
        interpret_clusterisation(dataset_to_interpret = tsne_agglomerative_dataset, result_dataset_name = "tsne_agglomerative_dataset.csv",
                                dataset_folder = path_to_tsne_2d_agglomerative_result)    

    # Interpretation for dataset reduced by linear pca algorithm
    if interpret_pca_reduced_data:
        # Gaussian clustering
        interpret_clusterisation(dataset_to_interpret = pca_gaussian_dataset, result_dataset_name = "pca_gaussian_dataset.csv",
                                dataset_folder = path_to_pca_2d_gauss_result)
        # K-Means clustering
        interpret_clusterisation(dataset_to_interpret = pca_kmeans_dataset, result_dataset_name = "pca_kmeans_dataset.csv",
                                dataset_folder = path_to_pca_2d_kmeans_result)
        # Agglomerative clustering
        interpret_clusterisation(dataset_to_interpret = pca_agglomerative_dataset, result_dataset_name = "pca_agglomerative_dataset.csv",
                                dataset_folder = path_to_pca_2d_agglomerative_result)

    # Interpretation for dataset reduced by kernel pca algorithm
    if interpret_kernel_pca_reduced_data:
        # Gaussian clustering
        interpret_clusterisation(dataset_to_interpret = kernel_pca_gaussian_dataset, result_dataset_name = "kernel_pca_gaussian_dataset.csv",
                                dataset_folder = path_to_kernel_pca_2d_gauss_result)
        # K-Means clustering
        interpret_clusterisation(dataset_to_interpret = kernel_pca_kmeans_dataset, result_dataset_name = "kernel_pca_kmeans_dataset.csv",
                                dataset_folder = path_to_kernel_pca_2d_kmeans_result)
        # Agglomerative clustering
        interpret_clusterisation(dataset_to_interpret = kernel_pca_agglomerative_dataset, result_dataset_name = "kernel_pca_agglomerative_dataset.csv",
                                dataset_folder = path_to_kernel_pca_2d_agglomerative_result)
    print("Interpretation complete!") 
