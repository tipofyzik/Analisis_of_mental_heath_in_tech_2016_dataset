# Analisis of "Mental heath in tech 2016" dataset from kaggle

## 1. Task formulation
To be brief, there is a high-dimensional, complex survey, which was conducted amongst technology-oriented employees. Dataset has missing values and non-standardized textual inputs. The goal is to categorize participants based on their survey responses and create visualizations that would simplify the data complexity. Key characteristics should be preserved and each result cluster should be described, in the context of its main traits.

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

You can find explanation of each config parameter in the section "4.2. Config and launch files".  
Best setup solutions for this task you can find in the section "5. Results of the work".  

## 4. Implementation
This implementation has 9 classes, 1 config file to tune desirable settings, and 1 launch file, .
### 4.1. Classes
Each class is located in its own file, which is documented so that the reader can understand the purpose of each function. Here, the main purpose of each class is briefly described.

**· _GraphPlotter class_** has functions to plot and save images for different cases. There are functions to plot and save graphs in batches (the number of columns multiplied by the number of rows), 2d and 3d representation of the dataset obtained via t-SNE, Kernel PCA and PCA algorithms, etc.  
**· _WorkingDatasetInfo class_** gives the basic information about the original dataset. This is important for understanding the type of data and how to process the dataset.  
**· _DatasetAnalyzer class_** is responsible for analyzing and preprocessing dataset. It counts missing values for categorical columns and removes those that exceed the threshold for missing values. It also prepares columns that originally have a lot of answer but can be transformed so that they become more appropriate for the task.  
**· _TextFeatureExtractor class_** processes columns with textual responses. It extracts all the most important phrases in a column and replaces the text in each cell with the most important phrase found in it.  
**· _DataEncoder class_** encodes prepared dataset making it prepared for machine learning algorithms.  
**· _OptimalClusterFinder class_** applies different algorithms to dataset to find the appropriate number of clusters for the clustering process. It creates and saves graphs with k-elbow method, silhouette scores for k-means clustering, dendrograms for agglomerative clustering, and BIC and AIC scores for Gaussian mixture.  
**· _DimensionalityReducer class_** is designed to reduce dimensionality of the dataset so it can be displayed in 2 or 3 dimensional space. There are 3 dimensionality reduciton algorithms: (Linear) PCA, Kernel PCA, and t-SNE.  
**· _DataClusterer class_** clusters reduced in dimensionality data. There are 3 clustering algorithms: K-Means, Gaussian Mixture, and Agglomerative clustering.  
**· _ResultInterpreter class_** takes the results of clustering and interprets them by selecting the most important features it can detect. The following algorithms are used for feature selection: chi squared test, mutual information score, and random forest algorithm mixed with permutation feature importance algorithm.  

### 4.2. Config and launch files
**Config file** contains settings for different stages of data analysis and clustering. You can set up parameters for matplotlib grid, image quality, number of clusters, random states for different algorithms, save paths, etc. There are 8 parameter categories in this .json file:  
**·         _DataPreprocessingParameters_:**  
**· _GraphPlotterGridParameters_:**  
**· _GraphPlotterSavePaths_:**  
**· _AdditionalParamters_:**  
**· _ClusteringParameters_:**  
**· _DimensionalityReducerParameters_:**  
**· _ResultInterpreterParameters_:**  
**· _ResultInterpreterSavePaths_:**  

### 4.3. Special tricks
There are 2 such columns: age and gender. Originally age column has answers distributed by year but certain age can be replaced with age range. 

### 4.4. How it works
After feature selection, **GraphPlotter class** displays and saves the response distribution among different classes for each top feature algorithms detected. Now, it can be 

## 5. Results of the work

## 6. Possible improvements
