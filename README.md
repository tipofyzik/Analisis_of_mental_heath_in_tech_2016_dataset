# Analisis of "Mental heath in tech 2016" dataset from kaggle

## 1. Task formulation
To be brief, there is a high-dimensional, complex survey, which was conducted amongst technology-oriented employees. Dataset has missing values and non-standardized textual inputs. The goal is to categorize participants based on their survey responses and create visualizations that would simplify the data complexity. Key characteristics should be preserved and main traits of each result cluster should be described.

Dataset can be accessed via this link: https://www.kaggle.com/datasets/osmi/mental-health-in-tech-2016  
In case this dataset would be changed, I uploaded it on this github page (**folder "code" -> folder "dataset"**).  

The results of the entire work are compressed into .zip file and can be accessed via this link: https://drive.google.com/drive/folders/1iFWibCaSU_GtRCDSgglztju0zFtNrz-4?usp=sharing  

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
&emsp;&emsp; — _with_text_columns_: This parameter is responsible for including columns with textual information in encoding and clustering process. If this parameter is set to 0 these columns won't be included. Otherwise, they will be encoded as others and will affect clustering results (I recommend setting 1 to avoid confusion).  
&emsp;&emsp; — _interpret_tsne_reduced_data_: This parameter determines whether the interpretation of clustering results for data reduced using the t-SNE algorithm is performed. If set to 0, no interpretation will be performed. If set to a non-zero value, the interpretation will be saved (I recommend setting 1 to avoid confusion).  
&emsp;&emsp; — _interpret_pca_reduced_data_: This parameter determines whether the interpretation of clustering results for data reduced using the PCA algorithm is performed. If set to 0, no interpretation will be performed. If set to a non-zero value, the interpretation will be saved (I recommend setting 1 to avoid confusion).  
&emsp;&emsp; — _interpret_kernel_pca_reduced_data_: This parameter determines whether the interpretation of clustering results for data reduced using the Kernel PCA algorithm is performed. If set to 0, no interpretation will be performed. If set to a non-zero value, the interpretation will be saved (I recommend setting 1 to avoid confusion).  
&emsp;&emsp; — _interpret_mds_reduced_data_: This parameter determines whether the interpretation of clustering results for data reduced using the MDS algorithm is performed. If set to 0, no interpretation will be performed. If set to a non-zero value, the interpretation will be saved (I recommend setting 1 to avoid confusion).  
**· _DimensionalityReducerParameters_:**  
&emsp;&emsp; — _save_info_ratio_: Contains the ratio for the Linear PCA method. You can specify either a fractional number representing the percentage of information retained or an integer indicating the number of features to keep.  
&emsp;&emsp; — _pca_random_state_: Contains the random state parameter for PCA algorithm. Serves to ensure reproducibility of results.  
&emsp;&emsp; — _kpca_kernel_: Contains the name of kernel function for Kernel PCA algorithm.  
&emsp;&emsp; — _kernel_pca_random_state_: Contains the random state parameter for Kernel PCA algorithm. Serves to ensure reproducibility of results.  
&emsp;&emsp; — _tsne_perplexity_: Contains the perplexity parameter for t-SNE algorithms.  
&emsp;&emsp; — _tsne_random_state_: Contains the random state parameter for t-SNE algorithm. Serves to ensure reproducibility of results.  
&emsp;&emsp; — _mds_random_state_: Contains the random state parameter for MDS algorithm. Serves to ensure reproducibility of results.  
&emsp;&emsp; — _tsne_slice_range_: The range of values for a specific component used to filter the data displayed in the 3D scatter plot for t-SNE visualization. Required to view data in a cross-section.  
&emsp;&emsp; — _linear_pca_slice_range_: The range of values for a specific component used to filter the data displayed in the 3D scatter plot for (Linear) PCA visualization. Required to view data in a cross-section.  
&emsp;&emsp; — _kernel_pca_slice_range_: The range of values for a specific component used to filter the data displayed in the 3D scatter plot for Kernel PCA visualization. Required to view data in a cross-section.  
&emsp;&emsp; — _mds_slice_range_: The range of values for a specific component used to filter the data displayed in the 3D scatter plot for MDS visualization. Required to view data in a cross-section.  
**· _ClusteringParameters_:**  
&emsp;&emsp; — _cluster_number_: Contains the number of cluster into which the data should be divided.  
&emsp;&emsp; — _kmeans_init_: Contains the method of initialization for K-Means algorithm.  
&emsp;&emsp; — _kmeans_random_state_: Contains the random state parameter for K-Means clustering algorithm. Serves to ensure reproducibility of results.  
&emsp;&emsp; — _gauss_random_state_:  Contains the random state parameter for Gaussian Mixture clustering algorithm. Serves to ensure reproducibility of results.  
**· _ResultInterpreterParameters_:**  
&emsp;&emsp; — _random_forest_with_permutations_:  This parameter controls the application of the permutation feature importance algorithm during Random Forest clustering. If this parameter is set to 0, the algorithm won't be applied; otherwise, it will be.  
&emsp;&emsp; — _random_forest_permutation_repeats_: Contains the number of repeats for permutation feature importance algorithm.  
&emsp;&emsp; — _permutation_random_state_: Contains the random state parameter for permutation feature importance algorithm. Serves to ensure reproducibility of results.  
&emsp;&emsp; — _mutual_information_random_state_: Contains the random state parameter for mutual information algorithm. Serves to ensure reproducibility of results.  
&emsp;&emsp; — _most_important_features_max_number_: Specifies the maximum number of most important features to display.  
**· _ResultInterpreterSavePaths_:**  
&emsp;&emsp; — _path_to_pca_2d_gauss_result_: Here, the output of the Gaussian Mixture clustering, performed on data after 2D reduction using Linear PCA, is saved.  
&emsp;&emsp; — _path_to_kernel_pca_2d_gauss_result_: Here, the output of the Gaussian Mixture clustering, performed on data after 2D reduction using Kernel PCA, is saved.  
&emsp;&emsp; — _path_to_tsne_2d_gauss_result_: Here, the output of the Gaussian Mixture clustering, performed on data after 2D reduction using t-SNE, is saved.  
&emsp;&emsp; — _path_to_mds_2d_gauss_result_: Here, the output of the Gaussian Mixture clustering, performed on data after 2D reduction using MDS, is saved.  
&emsp;&emsp; — _path_to_pca_2d_kmeans_result_: Here, the output of the K-Means clustering, performed on data after 2D reduction using Linear PCA, is saved.  
&emsp;&emsp; — _path_to_kernel_pca_2d_kmeans_result_: Here, the output of the K-Means clustering, performed on data after 2D reduction using Kernel PCA, is saved.  
&emsp;&emsp; — _path_to_tsne_2d_kmeans_result_: Here, the output of the K-Means clustering, performed on data after 2D reduction using t-SNE, is saved.  
&emsp;&emsp; — _path_to_mds_2d_kmeans_result_: Here, the output of the K-Means clustering, performed on data after 2D reduction using MDS, is saved.  
&emsp;&emsp; — _path_to_pca_2d_agglomerative_result_: Here, the output of the Agglomerative clustering, performed on data after 2D reduction using Linear PCA, is saved.  
&emsp;&emsp; — _path_to_kernel_pca_2d_agglomerative_result_: Here, the output of the Agglomerative clustering, performed on data after 2D reduction using Kernel PCA, is saved.  
&emsp;&emsp; — _path_to_tsne_2d_agglomerative_result_: Here, the output of the Agglomerative clustering, performed on data after 2D reduction using t-SNE, is saved.  
&emsp;&emsp; — _path_to_mds_2d_agglomerative_result_: Here, the output of the Agglomerative clustering, performed on data after 2D reduction using MDS, is saved.  

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
1. During the **data preparation** columns where the percentage of missed values exceeded the threshold have beem removed. Additionally, two columns were changed so that they become more suitable for analysis. There are two such columns: age and gender. In the age column, each cell contained individual response but have been replaced with age ranges. On the figure below you can clearly see the distribution before the transformation and after it. The original distribution has 53 unique responses, while the preprocessed one has only 5.  
![image](https://github.com/user-attachments/assets/56e7809c-6c1a-4404-8942-dfc19cc324bb)  
The same applies to the column regarding gender identity, which originally had 71 unique values. Transgender individuals form a small group, and separating their individual responses makes their representation even smaller. To make this group more representative and suitable for processing, all unique values for transgenders were replaced with the term 'transgender'. Additionally, most distinct values with significant counts for males and females were replaced with 'male' and 'female', respectively.
![image](https://github.com/user-attachments/assets/7f6b1816-23a6-4c3c-a515-1e6cbef82f13)  
2. **Textual feature extraction** is quite complex. It's performed for columns which contain arbitrary textual responses. There are 5 such columns. In the feature extraction process, the Bag-of-Words (BoW) method is applied to three textual columns, while the TF-IDF method is used for the remaining two columns. Furthermore, I was looking not for words, but for meaningfull phrases, i.e., n-grams. Once n-grams are identified they go through the filtering process that removes redundant n-grams. An n-gram is referred to be redundant if it intersects with more than 50% of the already selected n-grams. With additional condition the "importance" of each n-gram is also taken into account. Importance defines by BoW and TF-IDF method.  
For textual columns containing information about work positions and diagnoses, I apply Bag-of-Words method. During the filtering process, the word importance is also considered. In contrast, for columns with responses to "Why or why not?" questions, I used the TF-IDF measure without factoring in the importance of n-grams, instead sorting the n-grams by the number of words. This suggests to exclude short n-grams that are already contained in the long ones. After filtering, each column is treated individually. For each cell in the textual columns, we identify all the n-grams it contains and replace the response with the most significant n-gram. **I assumed** this n-gram would represent the essence of the participant's response and this worked for me.  
These processing decisions were derived imperically by launching the program and observing the corresponding result. Mentioned processing approach demonstrated the best feature extraction ability. This approach has a drawback though. In some sentences, there are the same repeating words and algorithm detected them. As a result, some responses were replaced with pairs of repeating words (see the upper graphs):  
![49_52_questions_from_text_data](https://github.com/user-attachments/assets/299c79e9-4244-4322-8434-59bd5ab5d982)  
3. **Encoding process** was performed using two encoding technique: label encoding and one-hot encoding. Empirically, columns with textual data, information about US states and territories, and binary columns (those containing only two possible responses) are better suited for label encoding than for one-hot encoding. Using one-hot encoding for these columns can degrade the quality of subsequent steps that can be seen clearly seen on dimensionality reduction step, when the distribution become more messy and less interpretable.   
It is important to note that this effect primarily arises from columns containing textual and territorial information. Binary columns can be encoded using either method, but it is advisable to use label encoding for other types of columns. If both binary and textual or territorial columns are encoded with one-hot encoding, the results can change dramatically.  
Label encoding is applied only to binary columns:  
![norm_tsne_2d_binary](https://github.com/user-attachments/assets/f57fd45d-3f51-4d42-9ba5-e9e8d3ef31de)  
Label encoding is applied only to territorial columns:  
![norm_tsne_2d_specific](https://github.com/user-attachments/assets/6af41c55-546a-4409-8baa-501d06c333ec)  
Label encoding is applied only to textual columns:  
![norm_tsne_2d_text](https://github.com/user-attachments/assets/07fefa5e-0f2c-4640-a6c0-f5aa7a3de585)  
4. **Interpretation process**  involves three feature selection algorithms. Each algorithm selects features and outputs a list of features sorted by their importance score in descending order. The results are then ranked based on their position in the sorted list: the feature with the highest importance score is assigned rank 1, the next feature rank is 2, and so on, with the least important feature receiving the highest rank number. In other words, the more important the feature, the lower its rank number. After all ranks are assigned, the mean rank across the applied algorithms is calculated, and the final features are sorted by this average rank in descending order. After this procedure, participants response distributions for each result cluster are saved in plots which makes it easy to interpret the result clusters.   

**Further, all type of mentioned columns here are encoded using label encoding!**

## 5. Results of the work
### 5.1 Nature of data
Let's start the discussion with the **nature of data** obtained via dimeansionality reduction. We have data visualizations in both 2D and 3D space. Furthermore, we have "slices" of dataset in 3D space to look at them from different angles. We have 4 reduction methods: Linear PCA, Kernel PCA, t-SNE, and Multidimensional Scaling (MDS):  

**PCA 2D and 3D visualizations and 3D slice:**
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/ee43bcc5-b720-4d38-8ecd-570c306beeb4" style="max-width:100%; height:auto;" /></td>
    <td><img src="https://github.com/user-attachments/assets/c219f493-8a7e-4c16-be8a-4dc132e22da1" style="max-width:100%; height:auto;" /></td>
  </tr>
</table>  

**Kernel PCA 2D and 3D visualizations and 3D slice:**  
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/d2a5e52a-f472-489f-b4c4-c847f9efa717" style="max-width:100%; height:auto;" /></td>
    <td><img src="https://github.com/user-attachments/assets/0e90cb7b-c814-4002-b3e0-9f759d580208" style="max-width:100%; height:auto;" /></td>
  </tr>
</table> 
 
**t-SNE 2D and 3D visualizations and 3D slice:**  
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/e1c905dd-10e5-4a2c-81be-bb9a52fc4ed3" style="max-width:100%; height:auto;" /></td>
    <td><img src="https://github.com/user-attachments/assets/59ca08cd-5cf5-44d6-a1f1-47148db19c05" style="max-width:100%; height:auto;" /></td>
  </tr>
</table> 
 
**MDS 2D and 3D visualizations and 3D slice:**  
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/593bc4b6-5180-4840-9e94-db33324c35fe" style="max-width:100%; height:auto;" /></td>
    <td><img src="https://github.com/user-attachments/assets/8c08a163-2384-4c1a-951b-c0913c9b551d" style="max-width:100%; height:auto;" /></td>
  </tr>
</table> 

PCA is desighned for linearly structured data, while other algorithms are better suited for non-linearly distributed types of data. Furthermore, PCA and Kernel PCA algorithms preserve global structure, whereas t-SNE and MDS strive to save local structure of data. The results depend on the parameters we choose, e.g., t-SNE can better save global or local structure depending on the perplexity parameter[1]. The main reason in using various algorithms, which were desighned for different purposes, is to better understand the nature of the given data.  

From obtained two- and three-dimensional representations, we can conclude that the data has non-linear structure. Moreover, data structure keeps the same for both cases: when columns with textual features are included and when they're not. Duscussing t-SNE, MDS, and PCA results, we can see that all visualizations in two-dimensional space don't have any distinguishable clusters of data points, indicating that the data is dustributed approximately evenly. This statement is supported by "slices" of data obtained in three-dimensional space. We see small deviations in the results of PCA and MDS algorithms, but there are no serious differences comparing to t-SNE and Kernel PCA. We can note the following things:  
1. In 2D space, Linear PCA produces results similar to those obtained by t-SNE and MDS algorithms: the output distribution resembles an ellipse, with fuzzy boundaries for PCA and more clearly defined contours for t-SNE and MDS. Furthermore, there are no distinguishable groups of points or one cluster highly consentrated in some area (the result distribution looks even). Thus, the data don't have distinct clusters, but probably contain some features (columns) that cause the data to be spread.  
2. Kernel PCA result highly depend on the kernel function we choose. In the presented gifs, you can see Kernel PCA representations obtained using the Gaussian radius basis function (RBF) kernel. However, if we select the sigmoid kernel, the result becomes more similar to those obtained by t-SNE and MDS, but noisier at the same time:  
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/498feced-78ab-4f00-bcf8-d5534cae5e5f" style="max-width:100%; height:auto;" /></td>
    <td><img src="https://github.com/user-attachments/assets/85fa8cd3-f024-4392-ab80-3b0e82d33d57" style="max-width:100%; height:auto;" /></td>
  </tr>
</table>  

So, t-SNE, MDS, and Kernel PCA with sigmoid kernel give us roughly similar results: an approximately evenly distributed data in the form of ellipse in 2D space. In 3D space, PCA and Kernel PCA show mirrored distributions (reflected along the vertical axis), with a subset of the points deviating from the main cluster. Furthermore, PCA and Kernel PCA give us nosier representations than other reductioon algorithms. Each representation doesn't have distinguishable clusters and high-density spots. Therefore, density-based clustering algorithms, such as DBSCAN clustering, aren't suitable for this dataset. Furthermore, due to the uniformity of the data, we expect to obtain only a few clusters at best.  

### 5.2 Number of clusters
Now, we should determine the appropriate for this dataset number of clusters. We'll consider 2 cases: when columns with textual information is considered and when the are not taken into account. Let's go through each cluster evaluation algorithm (see the corresponding graphs below):  
1. K-Elbow method is one of the most prominent and simplest methods for evaluating cluster for K-Means clustering. However, the output graphs don't have a clear elbow point [3]. Therefore, this algorithm isn't suitable in our case.  
2. Silhouette score is also used to evaluate the number of clusters k for K-Means algorithm. There are clear results, from which we can derive that for K-Means clustering it's better to choose 2 or 3 clusters depending on whether textual columns are considered. When choosing the optimal number of clusters, we should consider not only the algorithm's average score but also the distribution of clister sizes [3]. When textial features are taken into account, one cluster often becomes significantly larger than the others for many suggested cluster numbers. Therefore, the best choice is k=2, as larger k values result in significantly imbalanced cluster sizes. When textual columns are omitted, the best choice is k=3 since this configuration gives us the most size-balanced result.  
3. Dendrograms were built to evaluate the number of clusters for agglomerative clustering. In both cases, there are 3 clearly distinguishable clusters. However, when textual columns are considered, we can also say that there are 4 clusters.  
4. Finally, BIC/AIC scores were used to find optimal number of clusters for Gaussian Mixture clustering algorithm. The appropriate number corresponds to the global minimum on BIC or AIC graph. The Bayesian Information Criterion penalizes models more severely than The Akaike Information Criterion. This is why, if one criterion isn't well-suited for the model, we can choose the other one. In this case, AIC demostrates poor results, since there is no clear minimum. In contrast, BIC have (almost) clear minima for both scenarios: when textual columns are included, the optimal cluster number is k=2 (with a score a bit lower than for k=3), and when textual features are omitted, k=3 is preferred.  

The results of cluster choosing algorithms:  
Cluster choice when **textual columns are considered**:  
![cluster choice, with text graphs](https://github.com/user-attachments/assets/5dcea2da-da47-420d-80af-2c6dacc59570)  
![cluster choice, with text](https://github.com/user-attachments/assets/05464f94-5c57-438f-adb9-441189be172a)  

Cluster choice when **textual columns aren't considered**:  
![cluster choice, without text graphs](https://github.com/user-attachments/assets/ca0f230c-14b1-4ee6-abd1-519fba6c357d)  
![cluster choice, without text](https://github.com/user-attachments/assets/a75a5f74-028a-49d1-972c-e0a57285fa1a)  

The resulting parameter choices are shown below:  
1. We take into account columns with textual responses and split the data into 2 clusters:  
```json
  "AdditionalParamters": {
     "with_text_columns": 1
  },
  "ClusteringParameters": {
     "cluster_number": 2,
      # Other parameters
  },
```
2. We omit textual columns and choose either 3 (or 2) clusters:  
```json
  "AdditionalParamters": {
      "with_text_columns": 0
  },
  "ClusteringParameters": {
      "cluster_number": 3,
      # Other parameters
  },
```

### 5.3 Final clusters and their interpretation
**"config.json"** file contains 4 parameter in "AdditionalParamters" section:  
```json
  "AdditionalParamters": {
      # Other parameters
      "interpret_tsne_reduced_data": 1,
      "interpret_pca_reduced_data": 0,
      "interpret_kernel_pca_reduced_data": 0,
      "interpret_mds_reduced_data": 0
  }
```  
These parameters are responsible for interpretation of clustering results for data which dimensionality was reduced by a specific algorithm (see "4.2. Config file" section). Since the results of all algorithms are similar, we will focus on clustering and cluster interpretations for data reduced by t-SNE algorithm. If you want to perform and save interpretations for data reduced by other algorithms, you can simply replace zeroes in the parameters above with other 1 or other number.  
Additionally, the clustering algorithm used also should be considered during the interpretation process. Different clustering algorithms divide dataset into clusters in distinct ways, meaning that for each clustering algorithm there are unique features which trigger this algorithm. As a result, the interpretations of the clustering outcomes vary depending on the algorithm used. 

**!Note:** We will consoder only the features which mean ranks are less than 10. Features with mean rank greater than 10 (even greater than 8-9, in some cases) mostly have participants' response distributions with no clear distinctions (see interpretations below). The information about how many participants were assighned to each cluster you can find in the end (far right columns) of corresponding dataset tables that are located in interpretation folders. The number of cluster in graphs correspond to the number of cluster in the table plus 1, i.e., cluster 0 in table corresponds to cluster 1 in graphs.  

Let's look at output cluster distributions, what features have been selected for these distributions, and what response distributions have determined clusters. For simplicity and clarity, the result response distributions for each top feature will be shown in plots, so everyone can click on them and see the difference between clusters:  
**5.3.1 With textual columns (features), 2 clusters:**  
**K-Means clustering results:**  
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/d6aab37a-f0d3-482d-baa2-0e1fd7846a84" style="max-width:100%; height:auto;" /></td>
    <td><img src="https://github.com/user-attachments/assets/2fe384f4-f118-4d3e-8a0f-cf9c210837dd" style="max-width:100%; height:auto;" /></td>
  </tr>
</table>  

There are 8 features with average rank lower than 10. The result response distributions among these features (columns):  
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/9b0773b9-1406-4caa-9825-789669ef3e60" style="max-width:100%; height:auto;" /></td>
    <td><img src="https://github.com/user-attachments/assets/350c1d5f-9229-4124-b23b-004571739333" style="max-width:100%; height:auto;" /></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/f1eb8a4d-83e9-4893-b0bb-7d904e06721d" style="max-width:100%; height:auto;" /></td>
    <td><img src="https://github.com/user-attachments/assets/9f69351f-4e3f-42fb-8c42-35405ddf4732" style="max-width:100%; height:auto;" /></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/579b4431-ac34-4142-990a-8a46c01fcc9c" style="max-width:100%; height:auto;" /></td>
    <td><img src="https://github.com/user-attachments/assets/98c4bee3-1927-45c2-9c61-d277ccd8ec5c" style="max-width:100%; height:auto;" /></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/534363f6-fd8c-4430-951a-121ab5fdbb35" style="max-width:100%; height:auto;" /></td>
    <td><img src="https://github.com/user-attachments/assets/c64296b6-105b-4333-a10e-7a0cb7de085d" style="max-width:100%; height:auto;" /></td>
  </tr>
</table> 

There are 2 clusters: **1st cluster with 669 participants** and **2nd cluster with 764 participants**. The major distinctions between them can be seen in the first top-8 features:  
— "If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?":  
&emsp;&emsp; 1 cluster: most participants think that this is not applicable to them.  
&emsp;&emsp; 2 cluster: participants feel like not treating mental health problems "often" or "sometimes" affect their work.  
— "Have you had a mental health disorder in the past?":  
&emsp;&emsp; 1 cluster: more than half of participants had not mental health disorder in the past and the other subgroups had or maybe had it.  
&emsp;&emsp; 2 cluster: almost everyone had mental health disorder in the past.  
— "If you have a mental health issue, do you feel that it interferes with your work when being treated effectively?":  
&emsp;&emsp; 1 cluster: most participants think that this is not applicable to them.  
&emsp;&emsp; 2 cluster: participants feel like not treating mental health problems "sometimes" or "rarely" affect their work.  
— "Do you currently have a mental health disorder?":  
&emsp;&emsp; 1 cluster: more than half of participants don't have a mental health disorder, while the majority of the remaining are not sure whether they have it.  
&emsp;&emsp; 2 cluster: most participants have a mental health disorder.   
— "Have you been diagnosed with a mental health condition by a medical professional?":  
&emsp;&emsp; 1 cluster: most participants haven't ever been diognised with a mental health condition by a medical professional.  
&emsp;&emsp; 2 cluster: most participants have been diognised with a mental health condition by a medical professional.  
— "Have you ever sought treatment for a mental health issue from a mental health professional?":  
&emsp;&emsp; 1 cluster: most participants have never sought for a mental health  issue from a mental health professional.  
&emsp;&emsp; 2 cluster: most participants tried to seek for a mental health issue from a mental health professional.  
— "Do you have a family history of mental illness?":  
&emsp;&emsp; 1 cluster: most participants either don't have a family history of mental illness or don't know about it, while the others have it.  
&emsp;&emsp; 2 cluster: most participants have a family history of mental illness, while the others either don't have or don't know about it.  
— "If so, what condition(s) were you diagnosed with?":  
&emsp;&emsp; 1 cluster: this group is predominantly diognised with only mood disorder.  
&emsp;&emsp; 2 cluster: this group is predominantly diognised with both mood and anxiety disorder.  

Summing up, the 1st group of participants are poorly awared about their mental health issues and most of people didn't have and haven't ever worked with them before. Furthermore, the majority claim that they don't have any mental health issue, while they were diognised with mood disoreders. In contrast, most in the second participants group know a lot about their mental health conditions, they had issues in the past and tried to treat them. Additionally, fot the first and the third questions, the majority of the 1st group claims that these question are not related to them (question start with "If you have a mental health issue" and participants answered "not applicapable to me"). This suggests that they do not believe they have mental health issues. In contrast, for the other group, there is a clear indication that effective treatment could help reduce the impact on their work, as the response "often" dramatically dicreased and the response "rarely", conversely, increased in their answers.  

**Gaussian Mixture clustering results:**  
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/02167c00-07a7-4b74-ba3d-b247493315a2" style="max-width:100%; height:auto;" /></td>
    <td><img src="https://github.com/user-attachments/assets/f8e829d3-b5f6-49f8-b4b5-36cc1c908b99" style="max-width:100%; height:auto;" /></td>
  </tr>
</table>  

There are 8 features with average rank lower than 10. The result response distributions among these features (columns):  
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/05fe0b21-7780-48d2-81bb-fd769c4d92a0" style="max-width:100%; height:auto;" /></td>
    <td><img src="https://github.com/user-attachments/assets/6d14b210-3526-4af5-a3d9-2e74dd99222c" style="max-width:100%; height:auto;" /></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/5c32e217-12ff-4e57-9ca9-d67334be6589" style="max-width:100%; height:auto;" /></td>
    <td><img src="https://github.com/user-attachments/assets/52acc979-276f-40ab-9fe9-909a5a2e6677" style="max-width:100%; height:auto;" /></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/567eb97a-4548-4867-b7d9-41e7f730ee25" style="max-width:100%; height:auto;" /></td>
    <td><img src="https://github.com/user-attachments/assets/0b7c7217-cf85-4e1b-82d7-50d41abd4885" style="max-width:100%; height:auto;" /></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/c3712919-650c-4205-bf30-b14eb5b56ffa" style="max-width:100%; height:auto;" /></td>
    <td><img src="https://github.com/user-attachments/assets/6912821f-e147-4324-b8e4-a760fb7b161d" style="max-width:100%; height:auto;" /></td>
  </tr>
</table> 

There are 2 clusters: **1st cluster with 641 participants** and **2nd cluster with 792 participants**. The major distinctions between them can be seen in the first top-8 features:  
— "If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?"  
— "If you have a mental health issue, do you feel that it interferes with your work when being treated effectively?"  
— "Have you had a mental health disorder in the past?"  
— "Have you ever sought treatment for a mental health issue from a mental health professional?"  
— "Do you currently have a mental health disorder?"  
— "Have you been diagnosed with a mental health condition by a medical professional?"  
— "Do you have a family history of mental illness?"  
— "If so, what condition(s) were you diagnosed with?"  

This clustering method produce **almost the same results** like the previous one; only the order of the selected features and distributions for the last question have changed. In the previous clustering, most of the 1st participants group were diognised with mood disorders and the 2nd group had either a mood or an anxiety disorder. Now, the 1st group mostly have anxiety disorder, while the majority of the 2nd group have mood disorder and some people have anxiety disorder.     


**Agglomerative clustering results:**  
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/1fba01d4-b899-4ff3-8960-d4d256ee2930" style="max-width:100%; height:auto;" /></td>
    <td><img src="https://github.com/user-attachments/assets/d78092d0-dc98-4303-82f1-e06b911e7b2d" style="max-width:100%; height:auto;" /></td>
  </tr>
</table>  

There are 4 features with average rank lower than 10. The result response distributions among these features (columns):  
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/cb438f10-3f99-4d68-8bb1-860734341aca" style="max-width:100%; height:auto;" /></td>
    <td><img src="https://github.com/user-attachments/assets/107bb0e3-b199-4d6d-8874-aae7fadfa6f6" style="max-width:100%; height:auto;" /></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/d4daecb8-b1fb-43e3-8827-94e7204b9eb0" style="max-width:100%; height:auto;" /></td>
    <td><img src="https://github.com/user-attachments/assets/598497df-722c-4b79-8412-991d9b442e92" style="max-width:100%; height:auto;" /></td>
  </tr>
</table> 

There are 2 clusters: **1st cluster with 653 participants** and **2nd cluster with 780 participants**. The major distinctions between them can be seen in the first top-4 features:  
— "If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?":  
&emsp;&emsp; 1 cluster: participants feel like not treating mental health problems "often" or "sometimes" affect their work.  
&emsp;&emsp; 2 cluster: most participants think that this is not applicable to them.  
— "Have you had a mental health disorder in the past?":  
&emsp;&emsp; 1 cluster: almost everyone had mental health disorder in the past.  
&emsp;&emsp; 2 cluster: more than half of participants had not mental health disorder in the past and the other subgroups had or maybe had it.  
— "If you have a mental health issue, do you feel that it interferes with your work when being treated effectively?":  
&emsp;&emsp; 1 cluster: participants feel like not treating mental health problems "sometimes" or "rarely" affect their work.  
&emsp;&emsp; 2 cluster: most participants think that this is not applicable to them.  
— "Do you currently have a mental health disorder?":  
&emsp;&emsp; 1 cluster: most participants have a mental health disorder.   
&emsp;&emsp; 2 cluster: more than half of participants don't have a mental health disorder, while the majority of the remaining are not sure whether they have it.  
The features selected for this clustering method correspond to the first 4 features of the K-Means clistering. There is only one distinction between feature results: the class traits are swapped. Traits that belonged to the first class in K-Means clustering now pertain to the second cluster in Agglomerative clustering. The same applies to the second and the first clusters in K-Means and Agglomerative clustering, respectively. In other words, the clusters are the same, but their traits have been swapped.  

**Conclusions:**  
For the data that includes textual columns, we obtained 2 distinctive clusters (groups of people). Moreover, these clusters are the same for all clustering algorithms. The 1st group predominantly demonstrates ignorance about their mental health issues. They haven't ever been diognised with a mental health condition by a medical proffesional and haven't ever sought the treatment. Additionally, the majority of this group don't think that they don't have any mental health issues even though they were diognised with mood disorders. The other group have the opposite situaltion: participants monitor their mental health. They aware about their mental health issues, the family history of mental illness, and at least tried to treat their issies. However, not only the mood disorders, but also the anxiety disorders are common among this group.  

**5.3.2 Without textual columns (features), 3 clusters:**  
**K-Means clustering results:**  
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/9eb2f371-0690-4834-85b9-af9c1db15194" style="max-width:100%; height:auto;" /></td>
    <td><img src="https://github.com/user-attachments/assets/7499d06d-0dd4-4332-913a-c9203c820cf8" style="max-width:100%; height:auto;" /></td>
  </tr>
</table>  

There are 6 features with average rank lower than 10. The result response distributions among these features (columns):  
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/d506c32f-f35a-4037-8c7c-0a641a67bc42" style="max-width:100%; height:auto;" /></td>
    <td><img src="https://github.com/user-attachments/assets/fd691f58-127f-4519-87ec-bace659f7437" style="max-width:100%; height:auto;" /></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/64f5c7a0-2818-4b26-a95d-7536953b6f6e" style="max-width:100%; height:auto;" /></td>
    <td><img src="https://github.com/user-attachments/assets/bffb24f4-2535-4afc-b8b8-c0ec7c31d24a" style="max-width:100%; height:auto;" /></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/66f00219-bfb7-4463-82f2-191a3a7a7f4e" style="max-width:100%; height:auto;" /></td>
    <td><img src="https://github.com/user-attachments/assets/078a2a88-f537-4173-b056-36e88094f00e" style="max-width:100%; height:auto;" /></td>
  </tr>
</table> 

There are 3 clusters: **1st cluster with 613 participants**, **2nd cluster with 348 participants**, and **3nd cluster with 472 participants**. The major distinctions between them can be seen in the first top-6 features:  
— "Have you had a mental health disorder in the past?":  
&emsp;&emsp; 1 cluster: more than half of participants had not mental health disorder in the past and the others mostly didn't have it.  
&emsp;&emsp; 2 cluster: almost everyone had or maybe had mental health disorder in the past.  
&emsp;&emsp; 3 cluster: almost everyone had or maybe had mental health disorder in the past.  
— "If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?":  
&emsp;&emsp; 1 cluster: most participants think that this is not applicable to them.  
&emsp;&emsp; 2 cluster: participants feel like not treating mental health problems "often" or "sometimes" affect their work.  
&emsp;&emsp; 3 cluster: participants feel like not treating mental health problems "often" or "sometimes" affect their work.  
— "Have you ever sought treatment for a mental health issue from a mental health professional?":  
&emsp;&emsp; 1 cluster: most participants have never sought for a mental health  issue from a mental health professional.  
&emsp;&emsp; 2 cluster: most participants tried to seek for a mental health issue from a mental health professional.  
&emsp;&emsp; 3 cluster: most participants tried to seek for a mental health issue from a mental health professional.  
— "Do you currently have a mental health disorder?":  
&emsp;&emsp; 1 cluster: more than half of participants don't have a mental health disorder, while the majority of the remaining are not sure (due to the resonse "maybe") whether they have it.  
&emsp;&emsp; 2 cluster: most participants have or maybe have a mental health disorder.  
&emsp;&emsp; 3 cluster: most participants have or maybe have a mental health disorder.  
— "Did your previous employers provide resources to learn more about mental health issues and how to seek help?":  
&emsp;&emsp; 1 cluster: most participants haven't been provided with resourses about mental health issues on previous workplaces.  
&emsp;&emsp; 2 cluster: for most participants, there were some previous employers who provided them with resourses about mental health. However, there are participants who have never been provided with them.  
&emsp;&emsp; 3 cluster: most participants haven't been provided with resourses about mental health issues on previous workplaces.  
— "Do you feel that being identified as a person with a mental health issue would hurt your career?":  
&emsp;&emsp; 1 cluster: most participants think that this can hurt their career (corresponding responses are "maybe" and "yes, i think it would").  
&emsp;&emsp; 2 cluster: most participants think that this can hurt their career (corresponding responses are "maybe" and "yes, i think it would"). But some people in this group don't think so.  
&emsp;&emsp; 3 cluster: most participants think that this can hurt their career (corresponding responses are "maybe" and "yes, i think it would"). Additionally, some people think that it will definitely hurt their career.  

Again, the 1st group knows a little about their mental health issues. Most participants either didn't have mental health disorders in the past or aren't about that. Additionally, they don't think that they have any mental health issue for now. In contrast, the 2nd and the 3rd groups are more awared about their health issues. However, resourses to learn about mental health were provided only for the 2nd group by some of their previous employeers, while the other groups haven't been provided with anything. The only thing that unites these groups of people is that they all believe that "being identified as a person with a mental health issue" might hurt their career. And only the small subgroups think that it can't or would definitely affect them.   

**Gaussian Mixture clustering results:**  
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/c4f912e2-08fd-4a26-8b4f-1ff64e8b26b7" style="max-width:100%; height:auto;" /></td>
    <td><img src="https://github.com/user-attachments/assets/19e5e3c1-69ab-4183-8b23-39542b7f4c86" style="max-width:100%; height:auto;" /></td>
  </tr>
</table>  

There are 8 features with average rank lower than 10. The result response distributions among these features (columns):  
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/a28d2264-7ef2-4b8d-a293-e75c61ffc694" style="max-width:100%; height:auto;" /></td>
    <td><img src="https://github.com/user-attachments/assets/ea2d5814-847b-4912-967d-db3239896ee9" style="max-width:100%; height:auto;" /></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/98a13662-306a-449d-8036-9c4a1d32834f" style="max-width:100%; height:auto;" /></td>
    <td><img src="https://github.com/user-attachments/assets/3ffd2075-e60a-46f9-abe5-39014bff90f3" style="max-width:100%; height:auto;" /></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/b1f45feb-24a3-4352-ad45-43600786b4b2" style="max-width:100%; height:auto;" /></td>
    <td><img src="https://github.com/user-attachments/assets/2dfe8f26-adde-43e7-a31c-95033eaff124" style="max-width:100%; height:auto;" /></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/3870a300-7d72-49c3-8b81-d1ef112e041b" style="max-width:100%; height:auto;" /></td>
    <td><img src="https://github.com/user-attachments/assets/f9f92ee5-0241-42b1-9086-7dfa215c9d3a" style="max-width:100%; height:auto;" /></td>
  </tr>
</table> 

There are 3 clusters: **1st cluster with 622 participants**, **2nd cluster with 324 participants**, and **3nd cluster with 487 participants**. The major distinctions between them can be seen in the first top-8 features:  
**The same result as in the previous clustering method**  
— "Have you had a mental health disorder in the past?":  
— "If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?":  
— "Have you ever sought treatment for a mental health issue from a mental health professional?":  
— "Do you currently have a mental health disorder?":  
— "Do you feel that being identified as a person with a mental health issue would hurt your career?":  
— "Did your previous employers provide resources to learn more about mental health issues and how to seek help?":  
**Features unique to this clustering method**  
— "Have you been diagnosed with a mental health condition by a medical professional?":  
&emsp;&emsp; 1 cluster: most participants haven't ever been diognised with a mental health condition by a medical professional.  
&emsp;&emsp; 2 cluster: most participants have been diognised with a mental health condition by a medical professional.  
&emsp;&emsp; 3 cluster: most participants have been diognised with a mental health condition by a medical professional.  
— "Do you think that discussing a mental health disorder with your employer would have negative consequences?":  
&emsp;&emsp; 1 cluster: most participants think that it might or won't have negative consequences (corresponding responses are "maybe" and "no").  
&emsp;&emsp; 2 cluster: most participants think that it might or won't have negative consequences (corresponding responses are "maybe" and "no").  
&emsp;&emsp; 3 cluster: most participants think that it would or will definitely have negative consequences (corresponding responses are "maybe" and "yes").  

This clustering method produce **almost the same results** like the previous one; only the order of first 6 selected features has changed. However, there are 2 additional features that have mean rank lower than 10 (last two questions). Groups remained the same, so let's add some information about them via new features. People in the 1st group have never been diognised with a mental health issue by a medical professional and most of them don't think or aren't sure that discussion of the mental health disorder with their employers would have any negative consequences. On the other hand, people in the 2nd and the 3rd groupd were diognised by a medical prefessional. The second group shares the same opinion as the first group regarding the discussion of mental health disorders. However, the majority of the 3rd group either aren't sure or think that such a discussion can lead to some negative consequences.  

**Agglomerative clustering results:**  
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/58deb2fe-e408-4da8-a44c-8d9294b70921" style="max-width:100%; height:auto;" /></td>
    <td><img src="https://github.com/user-attachments/assets/9be15424-9c62-477d-8bde-602d8d0e7d2e" style="max-width:100%; height:auto;" /></td>
  </tr>
</table>  

There are 9 features with average rank lower than 10. The result response distributions among these features (columns):  
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/d6179ce9-cb29-48f4-bc7d-ebfd6da3dc64" style="max-width:100%; height:auto;" /></td>
    <td><img src="https://github.com/user-attachments/assets/e5f152ab-5a11-4151-a187-1ab24944e672" style="max-width:100%; height:auto;" /></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/05f6a283-bcf5-4461-bcf8-33ca099d6b1e" style="max-width:100%; height:auto;" /></td>
    <td><img src="https://github.com/user-attachments/assets/878f68e5-2d6e-49c0-b00a-176d8d302436" style="max-width:100%; height:auto;" /></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/664707b5-ad43-428f-a16c-3159dbe2e3da" style="max-width:100%; height:auto;" /></td>
    <td><img src="https://github.com/user-attachments/assets/ebbff9a4-5d78-40b6-82d9-06bb670bb68d" style="max-width:100%; height:auto;" /></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/555510b4-e938-4896-8c9a-c2f60c1382b7" style="max-width:100%; height:auto;" /></td>
    <td><img src="https://github.com/user-attachments/assets/57d0cf3b-6390-41be-a403-de48259037a9" style="max-width:100%; height:auto;" /></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/abd0e719-c2ba-42b0-823e-df5596b98a79" style="max-width:100%; height:auto;" /></td>
  </tr>
</table> 

There are 3 clusters: **1st cluster with 624 participants**, **2nd cluster with 552 participants**, and **3nd cluster with 257 participants**. The major distinctions between them can be seen in the first top-9 features:  
— "If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?":  
&emsp;&emsp; 1 cluster: most participants think that this is not applicable to them.  
&emsp;&emsp; 2 cluster: participants feel like not treating mental health problems "often" or "sometimes" affect their work.  
&emsp;&emsp; 3 cluster: participants feel like not treating mental health problems "often" or "sometimes" affect their work.  
— "Have you been diagnosed with a mental health condition by a medical professional?":  
&emsp;&emsp; 1 cluster: most participants haven't ever been diognised with a mental health condition by a medical professional.  
&emsp;&emsp; 2 cluster: most participants have been diognised with a mental health condition by a medical professional.  
&emsp;&emsp; 3 cluster: most participants have been diognised with a mental health condition by a medical professional.  
— "Do you feel that being identified as a person with a mental health issue would hurt your career?":  
&emsp;&emsp; 1 cluster: most participants think that this can hurt their career (corresponding responses are "maybe" and "yes, i think it would").  
&emsp;&emsp; 2 cluster: most participants think that this can hurt their career (corresponding responses are "maybe" and "yes, i think it would"). Additionally, some people think that it will definitely hurt their career.  
&emsp;&emsp; 3 cluster: most participants think that this can hurt their career (corresponding responses are "maybe" and "yes, i think it would"). But some people in this group don't think so.  
— "Do you currently have a mental health disorder?":  
&emsp;&emsp; 1 cluster: more than half of participants don't have a mental health disorder, while the majority of the remaining are not sure (due to the resonse "maybe") whether they have it.  
&emsp;&emsp; 2 cluster: most participants have or maybe have a mental health disorder.  
&emsp;&emsp; 3 cluster: most participants have or maybe have a mental health disorder.  
— "Did your previous employers provide resources to learn more about mental health issues and how to seek help?":  
&emsp;&emsp; 1 cluster: most participants haven't been provided with resourses about mental health issues on previous workplaces.  
&emsp;&emsp; 2 cluster: most participants haven't been provided with resourses about mental health issues on previous workplaces.  
&emsp;&emsp; 3 cluster: for most participants, there were some previous employers who provided them with resourses about mental health. However, there are participants who have never been provided with them.  
— "Have you had a mental health disorder in the past?":  
&emsp;&emsp; 1 cluster: more than half of participants had not mental health disorder in the past and the others mostly didn't have it.  
&emsp;&emsp; 2 cluster: almost everyone had or maybe had mental health disorder in the past.  
&emsp;&emsp; 3 cluster: almost everyone had or maybe had mental health disorder in the past.  
— "Do you think that discussing a mental health disorder with your employer would have negative consequences?":  
&emsp;&emsp; 1 cluster: most participants think that it might or won't have negative consequences (corresponding responses are "maybe" and "no").  
&emsp;&emsp; 2 cluster: most participants think that it might or will definitely have negative consequences (corresponding responses are "maybe" and "yes").  
&emsp;&emsp; 3 cluster: most participants think that it might or won't have negative consequences (corresponding responses are "maybe" and "no").  
— "If you have a mental health issue, do you feel that it interferes with your work when being treated effectively?":  
&emsp;&emsp; 1 cluster: most participants think that this is not applicable to them.  
&emsp;&emsp; 2 cluster: participants feel like not treating mental health problems "sometimes" or "rarely" affect their work.  
&emsp;&emsp; 3 cluster: participants feel like not treating mental health problems "sometimes" or "rarely" affect their work.  
— "Have you ever sought treatment for a mental health issue from a mental health professional?":  
&emsp;&emsp; 1 cluster: most participants have never sought for a mental health  issue from a mental health professional.  
&emsp;&emsp; 2 cluster: most participants tried to seek for a mental health issue from a mental health professional.  
&emsp;&emsp; 3 cluster: most participants tried to seek for a mental health issue from a mental health professional.  

There is the only one unique feature for this clustering method: "If you have a mental health issue, do you feel that it interferes with your work when being treated effectively?" Other features just changed their order. Furthermore, in this clustering method, the traits of the 2nd and the 3rd clusters swapped as it was in the previous (5.3.1) section. But let's go through the last feature. The first group don't thint that this question relates to them, while the other groups believe that a proper treatement may affect their work less often comparing to the situation if mental health disorder is NOT treated effectively.  

**Conclusions:**  
For the data that excludes textual columns, we obtained 3 distinctive clusters (groups of people). Moreover, these clusters are the same for all clustering algorithms. The 1st group predominantly demonstrates ignorance about their mental health issues as it was in the previous (5.3.1) section. They haven't ever been diognised with a mental health condition by a medical proffesional and haven't ever sought the treatment. Additionally, the majority of this group don't think that they don't have any mental health issues. The other groups have the opposite situaltion: participants of each group monitor their mental health. It's important to note that the 2nd and the 3rd group are very similar and their differences are not very clear. While in one group some minority of people think that "being identified as a person with a mental health issue" can hurt their career, the minority of the other group believe that it cannot. The same applies to other dissimilarities: groups are distinguished by the responses of minorities within them.  

## 6. Possible improvements
All ideas came into my mind after I finished and interpreted the project, so they aren't implemented. There are few things that can be improved and considered in the future works:  
1. During the textual feature extraction process, some information have been lost. We can see it on graphs in the section "4.4. Implementation specifics". Furthermore, extracted n-grams contain from 2 to 3 words which leads to the additional information loss. For example, if we perform feature extraction with n-grams' length from 1 to 3, we can find that the "stigma" is the main purpose in the "Why or why not" questions. **However, tests demonstrated that this effect barely affects the clustering output and interpretation results, so we can be no worried about this issue.**
<table>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/d2f48afb-603f-4694-b0c6-dfc7fc95c9a3" style="max-width:100%; height:auto;" />
      <p>Response distribution for n-grams with len from 1 to 3 ("Why or why not?" questions)</p>
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/7909e0bd-c05a-4805-b45f-c66d96396c44" style="max-width:100%; height:auto;" />
      <p>Reslut top feature for K-Means clustering</p>
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/1368f71d-4238-4f7a-b6dd-a4242671a90a" style="max-width:100%; height:auto;" />
      <p>Reslut top feature for Gaussian Mixture clustering</p>
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/29ab2079-98d7-4b3b-9e95-7b82ae6e832e" style="max-width:100%; height:auto;" />
      <p>Reslut top feature for Agglomerative clustering</p>
    </td>
  </tr>
</table>  

If you'd like to check it yourself, open the "TextFeatureExtractor.py" file, find the "__define_frequent_words_in_column" function and change "ngram range" in else section from (2,3) to (1,3):  
![image](https://github.com/user-attachments/assets/6d907b39-1591-4a20-8dc1-0cc477df8881)  

2. There are a lot of categorical columns that have from 4 to 5 possible answers. Due to this, feature selection algorithm can mistakenly highlight a feature as important, while there are kinda similar responses that just swapped their places. For instance, look again at the response distribution of the qustion "Do you feel that being identified as a person with a mental health issue would hurt your career?" The first 2 answers, "maybe" and "yes, i think it would" just swapping their places for each cluster. However, the meaning of these responses can be interpreted as "i'm not sure" and "i'm not sure, but i'm inclined to say yes", respectively. These are answers with varying degrees of confidence. For this reason, I assume that it would be good to try ordinal encoding for such questions instead of label encoding that was used in this work.
3. There are 3 cluster algorithm than were used in this task. To make the interpretation process less complex, a certain model can be chosen for clustering and further interpretation. For example BIC/AIC criterions can be used for model selection[5]  
4. To make graphs with response distributions easier to read and interpret, pie charts can be plotted instead of bars. It would help to see the percentage of participants that had chosen the certain answer.  

## 7. Worth to mention
1. In the archive with the results you can find the folder "2 clusters, without text". It contains the results for 2 clusters when textual columns aren't considered. These results weren't discussed here. The reason is why I didn't discuss them is that the output selected feature are the same to those we discussed in 5.3.1 and 5.3.2 section. There sis only one feature that become important due to different clustering results: "Were you aware of the options for mental health care provided by your previous employers?" The response distribution is predictible: the majority of the one group isn't awared about the options, while the other group was aware about some or all of them. Everything else is nearly idetical to results that we discussed in "5.3.1 With textual columns (features), 2 clusters" section.  
![7_FEAT~1](https://github.com/user-attachments/assets/2a55d67b-3cf0-4242-bc50-3158674ea386)
2. During the interpretation process I mainly mentioned top features that affected clustering the most (it's less than 10). However, there are other features (about 43) - shouldn't we consider them as well? There is the thing, response distribution for other features are quite similar. Yes, you can find that there are different numbers of participants for different clusters, however, there is no dramatic distinctions. If they were, algorithms should detected these features as major. That's why, if you'd like to see how data distributed among clusters for the remained features, I suggest you to do one of the following things:  
2.1 Firstly, you can open the archive I left (or the results folder if you run the program on your pc) and open the **"preprocessed_data_graphs"** folder. It's assumed that features with low importance score and, therefore, with a big mean rank have similar response distribution among all the output clusters.  
2.2 The second thing you can do is to set **"most_important_features_max_number"** parameter in **config.json** file to 53 (since there are maximim 53 features) and run the program. Once everithing is done open the **"interpretation_graphs"** folder, open the folder with results for clustering method you'd like and check the graphs in the corresponding **"ranked_top_53_features"** folder. You will find that there are all features sorted in the mean rank descending order. Check them and you'll find that difference is small, e.g., a couple of answers has a small difference and are swapped for different clusters.    




## 8. Literature  
[1] https://opentsne.readthedocs.io/en/latest/examples/03_preserving_global_structure/03_preserving_global_structure.html  
[2] https://scikit-learn.org/stable/auto_examples/manifold/plot_t_sne_perplexity.html  
[3] https://builtin.com/data-science/elbow-method  
[4] https://www.displayr.com/what-is-dendrogram/  
[5] https://censius.ai/blogs/machine-learning-model-selection-techniques  
