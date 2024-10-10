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
1. During the **data preparation** two columns were changed so that they become more suitable for analysis. There are two such columns: age and gender. In the age column, each cell contained individual response but have been replaced with age ranges. On the figure below you can clearly see the distribution before the transformation and after it. The original distribution has 53 unique responses, while the preprocessed one has only 5.  
![image](https://github.com/user-attachments/assets/56e7809c-6c1a-4404-8942-dfc19cc324bb)  
The same applies to the column regarding gender identity, which originally had 71 unique values. Transgender individuals form a small group, and separating their individual responses makes their representation even smaller. To make this group more representative and suitable for processing, all unique values for transgenders were replaced with the term 'transgender'. Additionally, most distinct values with significant counts for males and females were replaced with 'male' and 'female', respectively.
![image](https://github.com/user-attachments/assets/7f6b1816-23a6-4c3c-a515-1e6cbef82f13)  
2. **Textual feature extraction** is quite complex. In this process, the Bag-of-Words (BoW) method is applied to three columns, while the TF-IDF method is used for the remaining two columns. Furthermore, I was looking not for words, but for meaningfull phrases, i.e., n-grams. Once n-grams are identified they go through the filtering process that removes redundant n-grams. An n-gram is referred to be redundant if it intersects with more than 50% of the selected n-grams. With additional condition the "importance" of each n-gram is also taken into account. Importance defines by BoW and TF-IDF method.  
For textual columns containing information about work positions and diagnoses, I apply Bag-of-Words method. During the filtering process, the word importance is also considered. In contrast, for columns with responses to "Why or why not?" questions, I used the TF-IDF measure without factoring in the importance of n-grams, instead sorting the n-grams by the number of words. After filtering, each column is treated individually. For each cell in the textual columns, we identify all the n-grams it contains and replace the response with the most significant n-gram. **I assumed** this n-gram would represent the essence of the participant's response and this worked for me.  
These processing decisions were derived imperically by launching the program and observing the corresponding result. Mentioned processing approach demonstrated the best feature extraction ability. This approach has some drawback though. In some sentences there are the same repeating words and algorithm detected them. As a result, some responses were replaced with pairs of repeating words (see the upper graphs):  
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

**!Note:** We will consoder only the features which mean ranks are less than 10. Features with mean rank greater than 10 (even greater than 8-9, in some cases) mostly have participants' response distributions with no clear distinctions (see interpretations below).

Let's look at output cluster distributions, what features have been selected for these distributions, and what response distributions have determined clusters:  
**1. With textual columns (features), 2 clusters:**  
**1.1 K-Means clustering results:**  
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/d6aab37a-f0d3-482d-baa2-0e1fd7846a84" style="max-width:100%; height:auto;" /></td>
    <td><img src="https://github.com/user-attachments/assets/2fe384f4-f118-4d3e-8a0f-cf9c210837dd" style="max-width:100%; height:auto;" /></td>
  </tr>
</table>  



**1.2 Gaussian Mixture clustering results:**  
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/02167c00-07a7-4b74-ba3d-b247493315a2" style="max-width:100%; height:auto;" /></td>
    <td><img src="https://github.com/user-attachments/assets/f8e829d3-b5f6-49f8-b4b5-36cc1c908b99" style="max-width:100%; height:auto;" /></td>
  </tr>
</table>  



**1.3 Agglomerative clustering results:**  
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/1fba01d4-b899-4ff3-8960-d4d256ee2930" style="max-width:100%; height:auto;" /></td>
    <td><img src="https://github.com/user-attachments/assets/d78092d0-dc98-4303-82f1-e06b911e7b2d" style="max-width:100%; height:auto;" /></td>
  </tr>
</table>  





**2. Without textual columns (features), 3 clusters:**  
**2.1 K-Means clustering results:**  
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/9eb2f371-0690-4834-85b9-af9c1db15194" style="max-width:100%; height:auto;" /></td>
    <td><img src="https://github.com/user-attachments/assets/7499d06d-0dd4-4332-913a-c9203c820cf8" style="max-width:100%; height:auto;" /></td>
  </tr>
</table>  



**2.2 Gaussian Mixture clustering results:**  
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/c4f912e2-08fd-4a26-8b4f-1ff64e8b26b7" style="max-width:100%; height:auto;" /></td>
    <td><img src="https://github.com/user-attachments/assets/19e5e3c1-69ab-4183-8b23-39542b7f4c86" style="max-width:100%; height:auto;" /></td>
  </tr>
</table>  



**2.3 Agglomerative clustering results:**  
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/58deb2fe-e408-4da8-a44c-8d9294b70921" style="max-width:100%; height:auto;" /></td>
    <td><img src="https://github.com/user-attachments/assets/9be15424-9c62-477d-8bde-602d8d0e7d2e" style="max-width:100%; height:auto;" /></td>
  </tr>
</table>  




## 6. Possible improvements

## 7. Literature  
[1] https://opentsne.readthedocs.io/en/latest/examples/03_preserving_global_structure/03_preserving_global_structure.html  
[2] https://scikit-learn.org/stable/auto_examples/manifold/plot_t_sne_perplexity.html  
[3] https://builtin.com/data-science/elbow-method  
[4] https://www.displayr.com/what-is-dendrogram/  
