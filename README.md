# Analisis of "Mental heath in tech 2016" dataset from kaggle

## 1. Task formulation
To be brief, there is a high-dimensional, complex survey, which was conducted amongst technology-oriented employees. Dataset has missing values and non-standardized textual inputs. The goal is to categorize participants based on their survey responses and create visualizations that would simplify the data complexity. Key characteristics should be preserved and main traits of each result cluster should be described.

Dataset can be accessed via this link: https://www.kaggle.com/datasets/osmi/mental-health-in-tech-2016  
In case this dataset would be changed, I uploaded it on this github page (**folder "code" -> folder "dataset"**).  
## 2. Programm installation
### Requirements 
You need to intall Python with the version 3.11.3 and higher. All required modules to install you can find in the **"requirements.txt"** file.

### How to install
1. Download Python from the [official site](https://www.python.org/downloads/) and install it.  
2. Install required libraries from "requirements.txt".  
3. Download the folder "code" that contains the code to process dataset and the dataset itself. Run the **"app.py"** to process dataset and get results.  

## 3. How to use
There are two important files to run the programme: **"app.py"** and **"config.json"**. The first file runs the program, which includes preprocessing, encoding, clustering, and interpreting of the dataset. **"config.json"** allows to set up parameters for the program, e.g., number of clusters, random states for different algorithms, paths to save folders, etc.  

You can find explanation of each config parameter in the section "4.2. Config file".  
Best setup solutions for this task you can find in the section "5. Results of the work".  

## 4. Implementation
This implementation has 9 classes, 1 config file to tune desirable settings, and 1 launch file, .
### 4.1. Classes
Each class is located in its own file, which is documented so that the reader can understand the purpose of each function. Here, the main purpose of each class is briefly described.

**· _GraphPlotter class_** has functions to plot and save images for different cases. There are functions to plot and save graphs in batches (the number of columns multiplied by the number of rows), 2d and 3d representation of the dataset obtained via t-SNE, MDS, Kernel PCA, and PCA algorithms, etc.  
**· _WorkingDatasetInfo class_** gives the basic information about the original dataset. This is important for understanding the type of data and how to process the dataset.  
**· _DatasetAnalyzer class_** is responsible for analyzing and preprocessing dataset. It counts missing values for categorical columns and removes those that exceed the threshold for missing values. It also prepares columns that originally have a lot of answer but can be transformed so that they become more appropriate for the task.  
**· _TextFeatureExtractor class_** processes columns with textual responses. It extracts all the most important phrases in a column and replaces the text in each cell with the most important phrase found in it.  
**· _DataEncoder class_** encodes prepared dataset making it prepared for machine learning algorithms.  
**· _OptimalClusterFinder class_** applies different algorithms to dataset to find the appropriate number of clusters for the clustering process. It creates and saves graphs with k-elbow method, silhouette scores for k-means clustering, dendrograms for agglomerative clustering, and BIC and AIC scores for Gaussian mixture.  
**· _DimensionalityReducer class_** is designed to reduce dimensionality of the dataset so it can be displayed in 2 or 3 dimensional space. There are 3 dimensionality reduciton algorithms: (Linear) PCA, Kernel PCA, and t-SNE.  
**· _DataClusterer class_** clusters reduced in dimensionality data. There are 3 clustering algorithms: K-Means, Gaussian Mixture, and Agglomerative clustering.  
**· _ResultInterpreter class_** takes the results of clustering and interprets them by selecting the most important features it can detect. The following algorithms are used for feature selection: chi squared test, mutual information score, and random forest algorithm mixed with permutation feature importance algorithm.  

### 4.2. Config file
**Config file** contains settings for different stages of data analysis and clustering. You can set up parameters for matplotlib grid, image quality, number of clusters, random states for different algorithms, save paths, etc. There are 8 parameter categories in this .json file:  

**· _DataPreprocessingParameters_:**  
&emsp;&emsp; — _path_to_dataset_: Contains path to the original dataset that should be analyzed.  
&emsp;&emsp; — _missing_information_max_percent_: Contains the threshold that defines the maximum percentage of missing information allowed. Columns in which the percentage of empty cells exceeds this threshold will be removed (I refer to such columns as "**sparse columns**").  
**· _GraphPlotterGridParameters_:**  
&emsp;&emsp; — _n_columns_: Contains the number of columns that will be displayed on a matplotlib graph. This parameter is crucial for matplotlib subplot function. Each saved plot contains n_columns\*n_rows graphs.  
&emsp;&emsp; — _n_rows_: Contains the number of rows that will be displayed on a matplotlib graph. This parameter is crucial for matplotlib subplot function. Each saved plot contains n_columns\*n_rows graphs.  
&emsp;&emsp; — _dpi_: Contains dpi number to save images with a certain quality.   
&emsp;&emsp; — _max_feature_number_to_plot_: Contains the maximum feature (response) number that will be displayed on graphs.   
**· _GraphPlotterSavePaths_:**  
&emsp;&emsp; — _path_to_original_graphs_: Here, graphs with original dataset information are saved. Graphs shows response distribution per each question.   
&emsp;&emsp; — _path_to_processed_graphs_: Here, graphs with prepared for encoding dataset information (missed values are filled, text features extracted, and sparse columns are removed) are saved. Graphs shows response distribution per each retained question.  
&emsp;&emsp; — _path_to_cluster_choice_graphs_: Here, graphs with different metrics for choosing the number of clusters are saved. Contains folders from the following parameters: path_to_k_elbow, path_to_silhouette_score, path_to_dendrogram, path_to_bic_aic.  
&emsp;&emsp; — _path_to_k_elbow_: Here, the results of the K-Elbow algorithm's work are saved.  
&emsp;&emsp; — _path_to_silhouette_score_: Here, the results of the Silhouette score algorithm's work are saved.  
&emsp;&emsp; — _path_to_dendrogram_: Here, the the dendrograms are saved.  
&emsp;&emsp; — _path_to_bic_aic_: Here, graphs for the Bayesian Information Criterion (BIC) and Akaike Information Criterion (AIC) are saved.  
&emsp;&emsp; — _path_to_reduced_components_visualization_: Here, the results of various dimensionality reduction methods are saved. The results are presented in both 2D and 3D space.  
&emsp;&emsp; — _path_to_cluster_results_: Here, the results of diverse clustering methods are saved.  
&emsp;&emsp; — _path_to_interpretations_: Here, the results of the interpretation of the obtained clusters are saved. There are graphs showing the participants' responses for each top feature across the different clusters.  
**· _AdditionalParamters_:**  
&emsp;&emsp; — _with_text_columns_: This parameter is responsible for including columns with textual information in encoding and clustering process. If this parameter is set to 0 these columns won't be included. Otherwise, they will be encoded as others and will affect clustering results.   
**· _DimensionalityReducerParameters_:**  
&emsp;&emsp; — _save_info_ratio_: Contains the ratio for the Linear PCA method. You can specify either a fractional number representing the percentage of information retained or an integer indicating the number of features to keep.  
&emsp;&emsp; — _pca_random_state_: Contains the random state parameter for PCA algorithm. Serves to ensure reproducibility of results.  
&emsp;&emsp; — _kernel_pca_random_state_: Contains the random state parameter for Kernel PCA algorithm. Serves to ensure reproducibility of results.  
&emsp;&emsp; — _tsne_random_state_: Contains the random state parameter for t-SNE algorithm. Serves to ensure reproducibility of results.  
&emsp;&emsp; — _mds_random_state_: Contains the random state parameter for MDS algorithm. Serves to ensure reproducibility of results.  
&emsp;&emsp; — _tsne_slice_range_: 
&emsp;&emsp; — _linear_pca_slice_range_:   
&emsp;&emsp; — _kernel_pca_slice_range_:   
&emsp;&emsp; — _mds_slice_range_:   
**· _ClusteringParameters_:**  
&emsp;&emsp; — _cluster_number_: Contains the number of cluster into which the data should be divided.  
&emsp;&emsp; — _kmeans_init_: Contains the method of initialization for K-Means algorithm.  
&emsp;&emsp; — _kmeans_random_state_: Contains the random state parameter for K-Means clustering algorithm. Serves to ensure reproducibility of results.  
&emsp;&emsp; — _gauss_random_state_:  Contains the random state parameter for Gaussian Mixture clustering algorithm. Serves to ensure reproducibility of results.  
**· _ResultInterpreterParameters_:**  
&emsp;&emsp; — _random_forest_with_permutations_:  This parameter controls the application of the permutation feature importance algorithm during Random Forest clustering. If this parameter is set to 0, the algorithm won't be applied; otherwise, it will be.  
&emsp;&emsp; — _random_forest_permutation_repeats_: Contains the number of repeats for permutation feature importance algorithm.  
&emsp;&emsp; — _permutation_random_state_: Contains the random state parameter for permutation feature importance algorithm. Serves to ensure reproducibility of results.  
&emsp;&emsp; — _mutual_information_random_state_:   
&emsp;&emsp; — _most_important_features_max_number_:   
&emsp;&emsp; — _interpret_tsne_reduced_data_:   
&emsp;&emsp; — _interpret_pca_reduced_data_:   
&emsp;&emsp; — _interpret_kernel_pca_reduced_data_:   
**· _ResultInterpreterSavePaths_:**  
&emsp;&emsp; — _path_to_pca_2d_gauss_result_: Here, the output of the Gaussian Mixture clustering, performed on data after 2D reduction using Linear PCA, is saved.  
&emsp;&emsp; — _path_to_kernel_pca_2d_gauss_result_: Here, the output of the Gaussian Mixture clustering, performed on data after 2D reduction using Kernel PCA, is saved.  
&emsp;&emsp; — _path_to_tsne_2d_gauss_result_: Here, the output of the Gaussian Mixture clustering, performed on data after 2D reduction using t-SNE, is saved.  
&emsp;&emsp; — _path_to_pca_2d_kmeans_result_: Here, the output of the K-Means clustering, performed on data after 2D reduction using Linear PCA, is saved.  
&emsp;&emsp; — _path_to_kernel_pca_2d_kmeans_result_: Here, the output of the K-Means clustering, performed on data after 2D reduction using Kernel PCA, is saved.  
&emsp;&emsp; — _path_to_tsne_2d_kmeans_result_: Here, the output of the K-Means clustering, performed on data after 2D reduction using t-SNE, is saved.  
&emsp;&emsp; — _path_to_pca_2d_agglomerative_result_: Here, the output of the Agglomerative clustering, performed on data after 2D reduction using Linear PCA, is saved.  
&emsp;&emsp; — _path_to_kernel_pca_2d_agglomerative_result_: Here, the output of the Agglomerative clustering, performed on data after 2D reduction using Kernel PCA, is saved.  
&emsp;&emsp; — _path_to_tsne_2d_agglomerative_result_: Here, the output of the Agglomerative clustering, performed on data after 2D reduction using t-SNE, is saved.  

### 4.3. Launch file
All processes, from reading the dataset to clustering it, take place in the **"app.py"** file. Let's go through a step-by-step explanation of what happens there. The program:  
1. Imports all the custom classes and reads parameters from **"config.json"**.  
2. Reads the original dataset and outputs basic information about it to the console. Furthermore, for each dataset column, a graph showing the distribution of responses is plotted.  
3. Removes columns where the percentage of missing values exceeds the predefined threshold. Then, fills in missing values for retained columns.  
4. Extracts features for columns with textual data and replaces each complex textual responce in a cell with most important phrase that this text includes. For more information on textual feature extraction, see the "4.4. Implementation specifics" section. Now, our dataset is almost ready for machine learning algorithms.  
5. Plots and saves graphs of each column of the preprocessed dataset, similar to how it was done in step 2 for the original one.  
6. Encodes and normalizes prepared dataset.  
7. Applies algorithms to determine the optimal number of clusters and saves the results for each algorithm.  
8. Applies dimensionality reduction algorithms to visualize the given high-dimensional dataset in both 2D and 3D space. Saves plots with these visualizations.  
9. Performs clustering of the dataset using various clustering algorithms. Saves plots with clusterization results.  
10. Interprets the results of multiple clustering algorithms by selecting key features using various methods, ranking the top features for each method, calculating the average rank for each top feature across all methods, and ultimately visualizing the distribution of responses for each top feature across clusters.  

### 4.4. Implementation specifics
1. During the **data preparation** two columns were changed so that they become more suitable for analysis. There are two such columns: age and gender. In the age column, each cell contained individual response but have been replaced with age ranges. On the figure below you can clearly see the distribution before the transformation and after it. The original distribution has 53 unique responses, while the preprocessed one has only 5.  
![image](https://github.com/user-attachments/assets/56e7809c-6c1a-4404-8942-dfc19cc324bb)  
The same applies to the column regarding gender identity, which originally had 71 unique values. Transgender individuals form a small group, and separating their individual responses makes their representation even smaller. To make this group more representative and suitable for processing, all unique values for transgenders were replaced with the term 'transgender'. Additionally, most distinct values with significant counts for males and females were replaced with 'male' and 'female', respectively.
![image](https://github.com/user-attachments/assets/7f6b1816-23a6-4c3c-a515-1e6cbef82f13)  
2. **Textual feature extraction** is quite complex. In this process, the Bag-of-Words (BoW) method is applied to three columns, while the TF-IDF method is used for the remaining two columns. Furthermore, I was looking not for words, but for meaningfull phrases, i.e., n-grams. Once n-grams are identified they go through the filtering process that removes redundant n-grams. An n-gram is referred to be redundant if it intersects with more than 50% of the selected n-grams. With additional condition the "importance" of each n-gram is also taken into account. Importance defines by BoW and TF-IDF method.  
For textual columns containing information about work positions and diagnoses, I apply Bag-of-Words method. During the filtering process, the word importance is also considered. In contrast, for columns with responses to "Why or why not?" questions, I used the TF-IDF measure without factoring in the importance of n-grams, instead sorting the n-grams by the number of words. After filtering, each column is treated individually. For each cell in the textual columns, we identify all the n-grams it contains and replace the response with the most significant n-gram. **I assumed** this n-gram would represent the essence of the participant's response and this worked for me.  
These processing decisions were derived imperically by launching the program and observing the corresponding result. Mention processing approach demonstrated the best feature extraction ability.  
3. **Encoding process** was performed using two encoding technique: label encoding and one-hot encoding. Empirically, columns with textual data, information about US states and territories, and binary columns (those containing only two possible responses) are better suited for label encoding than for one-hot encoding. Using one-hot encoding for these columns can degrade the quality of subsequent steps that can be seen clearly seen on dimensionality reduction step, when the distribution become more messy and less interpretable.   
It is important to note that this effect primarily arises from columns containing textual and territorial information. Binary columns can be encoded using either method, but it is advisable to use label encoding for other types of columns. If both binary and textual or territorial columns are encoded with one-hot encoding, the results can change dramatically.

## 5. Results of the work
The results of the work are located here:   

Let's start the discussion with the **nature of data** obtained via dimeansionality reduction. We got data visualization in both 2D and 3D space. Furthermore, we have "slices" of dataset in 3D space to look at them from different angles. We have 3 reduction methods: Linear PCA, Kernel PCA, t-SNE, and Multidimensional Scaling (MDS):  
![linear_pca_slice](https://github.com/user-attachments/assets/e07a1fe7-dba5-46e8-b892-76f62d975e96)  
![kernel_pca_slice](https://github.com/user-attachments/assets/980dcf07-d282-435b-90a7-c99d5a20b8cd)  
![tsne_slice](https://github.com/user-attachments/assets/4c9a359e-f0a1-4b81-a580-c343a7025801)  
![mds_slice](https://github.com/user-attachments/assets/a623d7ad-1911-4cbc-845b-9834a93d60f1)  



Now, we should define the parameters that give us the best results. Firstly, look at the result of choosing cluster number algorithms:  
Cluster choice when **textual columns are considered**:  
![cluster choice, with text graphs](https://github.com/user-attachments/assets/9c0947e6-16c2-4b11-bd45-e325513c0a89)  
![cluster choice, with text](https://github.com/user-attachments/assets/05464f94-5c57-438f-adb9-441189be172a)  

Cluster choice when **textual columns aren't considered**:  
![cluster choice, without text graphs](https://github.com/user-attachments/assets/31730e36-8a35-44e3-8f9a-b95956057fe5)  
![cluster choice, without text](https://github.com/user-attachments/assets/a75a5f74-028a-49d1-972c-e0a57285fa1a)  


1. We take into account columns with textual responses and split the data into 2 clasters:  
   ```json
    "AdditionalParamters": {
        "with_text_columns": 1
    },
    "ClusteringParameters": {
        "cluster_number": 2,
        "kmeans_init": "k-means++",
        "kmeans_random_state": 0,
        "gauss_random_state": 0
    },
```
2. We omit textual columns and choose either 2 or 3 clusters (3 clusters below):  
```json
    "AdditionalParamters": {
        "with_text_columns": 0
    },
    "ClusteringParameters": {
        "cluster_number": 3,
        "kmeans_init": "k-means++",
        "kmeans_random_state": 0,
        "gauss_random_state": 0
    },
```

## 6. Possible improvements
