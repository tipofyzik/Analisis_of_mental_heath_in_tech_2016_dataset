from WorkingDatasetInfo import WorkingDatasetInfo
from DatasetAnalyzer import DatasetAnalyzer
from TextFeatureExtractor import TextFeatureExtractor
from GraphPlotter import GraphPlotter
from DataEncoder import DataEncoder
# from DataDecoder import DataDecoder
from OptimalClusterFinder import OptimalClusterFinder
from DimensionalityReducer import DimensionalityReducer
from DataClusterer import DataClusterer
from ResultInterpreter import ResultInterpreter

import pandas as pd
import json


with open('config.json', 'r') as f:
    config = json.load(f)

path_to_dataset = config['WorkingDataset']['path_to_dataset']
missing_information_max_percent = config['WorkingDataset']['missing_information_max_percent']
n_columns = config['GraphPlotter']['n_columns']
n_rows = config['GraphPlotter']['n_rows']
max_feature_number = config['GraphPlotter']['max_feature_number_to_plot']

path_to_original_graphs = config['GraphPlotter']['path_to_original_graphs']
path_to_processed_graphs = config['GraphPlotter']['path_to_processed_graphs']
path_to_cluster_choice_graphs = config['GraphPlotter']['path_to_cluster_choice_graphs']
path_to_k_elbow = config['GraphPlotter']['path_to_k_elbow']
path_to_silhouette_score = config['GraphPlotter']['path_to_silhouette_score']
path_to_dendrogram = config['GraphPlotter']['path_to_dendrogram']
path_to_reduced_components_visualization = config['GraphPlotter']['path_to_reduced_components_visualization']
path_to_cluster_results = config['GraphPlotter']['path_to_cluster_results']

save_info_ratio = config["DimensionalityReducer"]["save_info_ratio"]
tsne_random_state = config["DimensionalityReducer"]["tsne_random_state"]
kmeans_cluster_number = config["KMeansParameters"]["kmeans_cluster_number"]
kmeans_init = config["KMeansParameters"]["kmeans_init"]
kmeans_random_state = config["KMeansParameters"]["kmeans_random_state"]
gauss_random_state = config["GaussianMixtureParameters"]["gauss_random_state"]

path_to_interpretations = config['GraphPlotter']['path_to_interpretations']
path_to_gauss_result = config["ResultInterpreter"]["path_to_gauss_result"]
path_to_kmeans_result = config["ResultInterpreter"]["path_to_kmeans_result"]
path_to_agglomerative_result = config["ResultInterpreter"]["path_to_agglomerative_result"]
permutations_for_random_forest = bool(config["ResultInterpreter"]["permutations_for_random_forest"])

if __name__ == "__main__":
    dataset = pd.read_csv(path_to_dataset)
    dataset_info = WorkingDatasetInfo(dataset)
    analyzer = DatasetAnalyzer(dataset)
    plotter = GraphPlotter(n_columns, n_rows, max_feature_number)
    cluster_finder = OptimalClusterFinder(kmeans_init, kmeans_random_state)
    dimension_reducer = DimensionalityReducer(tsne_random_state)

    # Getting dataset basic info
    dataset_info.print_dataset_info()
    # dataset_info.print_each_column_types()
    analyzer.check_missing_values(percent_threshold=missing_information_max_percent)
    plotter.save_plots(path_to_original_graphs, dataset)
    print("Analysis complete!")

    # Analyzing and preparing data for future working
    analyzer.drop_sparse_columns()
    analyzer.preprocess_columns()
    categorical_dataset, text_dataset = analyzer.return_divided_datasets()
    feature_extractor = TextFeatureExtractor(text_dataset)
    feature_extractor.extract_features()
    print("Data preparation complete!")

    dataset = pd.concat([categorical_dataset, text_dataset], axis = 1)
    plotter.save_plots(path_to_processed_graphs, dataset)
    print("Graphs with prepared data saved!")

    # Encoding data for machine learning algorithms to work
    encoder = DataEncoder()
    encoder.pass_text_columns(text_dataset.columns)
    encoder.encode_data(dataset)
    encoder.normalize_data()
    encoded_dataset, normalized_dataset = encoder.get_encoded_dataset()
    print("Data encoded!")



    # Finding the optimal number of clusters
    def find_optimal_k(dataset: pd.DataFrame, type_of_dataset: str, dendrogram_threshold: int):
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



    # Reducing dimensionality
    pca_95_result = dimension_reducer.get_95_percent_pca_result(saved_info_ratio = 0.95, normalized_dataset = normalized_dataset)

    # Linear PCA
    norm_pca_2d_result, norm_pca_3d_result = dimension_reducer.\
        get_linear_pca_result(normalized_dataset = normalized_dataset)
    # Kernel PCA and t-SNE for normalized data
    norm_kernel_pca_2d_result, norm_kernel_pca_3d_result = dimension_reducer.\
        get_kernel_pca_result(normalized_or_pca_dataset = normalized_dataset)
    norm_tsne_2d_result, norm_tsne_3d_result = dimension_reducer.\
        get_tsne_result(normalized_or_pca_dataset = normalized_dataset)
    # Kernel PCA and t-SNE for data reduced by linear PCA with 95% information saved
    pca_kernel_pca_2d_result, pca_kernel_pca_3d_result = dimension_reducer.\
        get_kernel_pca_result(normalized_or_pca_dataset = pca_95_result)
    pca_tsne_2d_result, pca_tsne_3d_result = dimension_reducer.\
        get_tsne_result(normalized_or_pca_dataset = pca_95_result)

    find_optimal_k(dataset = encoded_dataset, type_of_dataset = "encoded", dendrogram_threshold = 60)
    find_optimal_k(dataset = normalized_dataset, type_of_dataset = "normalized", dendrogram_threshold = 62)
    find_optimal_k(dataset = pca_95_result, type_of_dataset = "pca", dendrogram_threshold = 65)
    print("Cluster number evaluation complete!\n")



    # Clusterization
    clusterer = DataClusterer(cluster_number = kmeans_cluster_number,
                              kmeans_random_state = kmeans_random_state,
                              kmeans_init = kmeans_init,
                              gauss_random_state = gauss_random_state)

    gaussian_labels = clusterer.gaussian_mixture_clusterization(norm_pca_2d_result)
    kmeans_cluster_labels = clusterer.kmeans_clusterizaiton(norm_tsne_2d_result)
    agglomerative_cluster_labels = clusterer.agglomerative_clusterization(norm_tsne_2d_result)

    gaussian_dataset = dataset.copy()
    gaussian_dataset['Cluster'] = gaussian_labels
    kmeans_dataset = dataset.copy()
    kmeans_dataset['Cluster'] = kmeans_cluster_labels
    agglomerative_dataset = dataset.copy()
    agglomerative_dataset['Cluster'] = agglomerative_cluster_labels

    # Save reduced data visualization
    plotter.save_2d_reduced_data_plotes(path_to_save = path_to_reduced_components_visualization, file_name = "linear_pca_2d.png",
                                  reducing_method="Linear_PCA_2D", reduced_data = norm_pca_2d_result)
    plotter.save_3d_reduced_data_plotes(path_to_save = path_to_reduced_components_visualization, file_name = "linear_pca_3d.png",
                                  reducing_method="Linear_PCA_3D", reduced_data = norm_pca_3d_result)

    plotter.save_2d_reduced_data_plotes(path_to_save = path_to_reduced_components_visualization, file_name = "kernel_pca_2d.png",
                                  reducing_method="Kernel_PCA_2D", reduced_data = norm_kernel_pca_2d_result)
    plotter.save_3d_reduced_data_plotes(path_to_save = path_to_reduced_components_visualization, file_name = "kernel_pca_3d.png",
                                  reducing_method="Kernel_PCA_3D", reduced_data = norm_kernel_pca_3d_result)
    
    plotter.save_2d_reduced_data_plotes(path_to_save = path_to_reduced_components_visualization, file_name = "tsne_2d.png",
                                  reducing_method="tSNE_2D", reduced_data = norm_tsne_2d_result)
    plotter.save_3d_reduced_data_plotes(path_to_save = path_to_reduced_components_visualization, file_name = "tsne_3d.png",
                                  reducing_method="tSNE_3D", reduced_data = norm_tsne_3d_result)
    
    
    # Save clustering results
    plotter.save_clustering_plots(path_to_save = path_to_cluster_results, file_name = "gaussian_on_pca.png", 
                                  type_of_clustering = "Gaussian", reducing_method="PCA", 
                                  reduced_data = norm_pca_2d_result, cluster_labels = gaussian_labels)
    plotter.save_clustering_plots(path_to_save = path_to_cluster_results, file_name = "kmeans_on_tsne.png", 
                                  type_of_clustering = "KMeans", reducing_method="t-SNE", 
                                  reduced_data = norm_tsne_2d_result, cluster_labels = kmeans_cluster_labels)
    plotter.save_clustering_plots(path_to_save = path_to_cluster_results, file_name = "agglomerative_on_tsne.png", 
                                  type_of_clustering = "Agglomerative", reducing_method="t-SNE", 
                                  reduced_data = norm_tsne_2d_result, cluster_labels = agglomerative_cluster_labels)
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
        plotter.save_chi_squared_plot(chi2_result = chi2_result, main_folder = path_to_interpretations, 
                                      dataset_folder = dataset_folder)
        plotter.plot_important_features(dataset = dataset_to_interpret, columns_to_plot = chi2_result["Feature"][:max_feature_number],
                                        main_folder = path_to_interpretations, dataset_folder = dataset_folder,
                                        test_name = "chi_squared_selected_features")
        #Mutual information
        mutual_info_result = interpreter.mutual_info_feature_selection(number_of_features=20)
        plotter.save_mutual_info_plot(mutual_info_result = mutual_info_result, main_folder = path_to_interpretations, 
                                      dataset_folder = dataset_folder)
        plotter.plot_important_features(dataset = dataset_to_interpret, columns_to_plot = mutual_info_result["Feature"][:max_feature_number],
                                        main_folder = path_to_interpretations, dataset_folder = dataset_folder,
                                        test_name = "mutual_information_selected_features")
        #Random Forest
        random_forest_result = interpreter.random_forest_feature_selection(with_permutations = permutations_for_random_forest)
        plotter.save_random_forest_plot(random_forest_result = random_forest_result, main_folder = path_to_interpretations, 
                                      dataset_folder = dataset_folder)
        plotter.plot_important_features(dataset = dataset_to_interpret, columns_to_plot = random_forest_result["Feature"][:max_feature_number],
                                        main_folder = path_to_interpretations, dataset_folder = dataset_folder,
                                        test_name = "random_forest_selected_features")
        
    interpret_clusterisation(dataset_to_interpret = gaussian_dataset, result_dataset_name = "gaussian_dataset.csv",
                             dataset_folder = path_to_gauss_result)
    interpret_clusterisation(dataset_to_interpret = kmeans_dataset, result_dataset_name = "kmeans_dataset.csv",
                             dataset_folder = path_to_kmeans_result)
    interpret_clusterisation(dataset_to_interpret = agglomerative_dataset, result_dataset_name = "agglomerative_dataset.csv",
                             dataset_folder = path_to_agglomerative_result)
    print("Interpretation complete!")