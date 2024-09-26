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

You can find explanation of each parameter in the section "4. Implementation".  
Best setup solutions for this task you can find in the section "5. Results of the work".  

## 4. Implementation
This implementation has 9 classes, 1 launch file, and 1 config file to tune desirable settings.
### 4.1. Classes
**· *GraphPlotter***:  
**· *WorkingDatasetInfo***:  
**· *DatasetAnalyzer***:  
**· *TextFeatureExtractor***:  
**· *DataEncoder***:  
**· *OptimalClusterFinder***:  
**· *DimensionalityReducer***:  
**· *DataClusterer***:  
**· *ResultInterpreter***:  
### 4.2. Config and launch files


## 5. Results of the work

## 6. Possible improvements
