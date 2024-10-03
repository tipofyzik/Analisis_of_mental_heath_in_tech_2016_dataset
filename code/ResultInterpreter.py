from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import chi2_contingency

from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os



class ResultInterpreter:
    """
    A class for interpreting clustering results from a dataset.

    Attributes:
        __path_to_folder (str): Path to the folder where the interpretation results will be saved.
        __encoded_dataset (pd.DataFrame): The dataset with encoded categorical features.
        __target_column (pd.Series): The target column containing cluster labels.
    """
    
    def __init__(self, dataset: pd.DataFrame, result_dataset_name: str,
                 main_folder: str, dataset_folder: str):
        """
        Initializes the ResultInterpreter object and saves the dataset to a file.
        Also preprocesses the dataset for interpretation by encoding categorical features.

        Args:
            dataset (pd.DataFrame): The dataset to interpret.
            result_dataset_name (str): The name of the file to save the encoded dataset.
            main_folder (str): The main folder path where results will be stored.
            dataset_folder (str): The subfolder path within the main folder where the dataset will be saved.
        """
        self.__path_to_folder = os.path.join(main_folder, dataset_folder)
        self.__save_dataset_to_file(dataset, result_dataset_name)
        self.__encoded_dataset, self.__target_column = self.__preprocess_for_interpretation(dataset)

    def __preprocess_for_interpretation(self, dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        Preprocesses the dataset by encoding categorical variables and separating features from the target.

        Args:
            dataset (pd.DataFrame): The dataset to preprocess.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Encoded feature matrix and target column.
        """
        label_encoders = {}
        X = dataset.drop(columns=['Cluster'])
        y = dataset['Cluster']
        
        for column in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column])
            label_encoders[column] = le
        return X, y
    
    def __save_dataset_to_file(self, dataset: pd.DataFrame, result_dataset_name: str) -> None:
        """
        Saves the dataset to a specified folder as a CSV file.

        Args:
            dataset (pd.DataFrame): The dataset to save.
            result_dataset_name (str): The name of the CSV file.
        """
        os.makedirs(self.__path_to_folder, exist_ok=True)
        dataset.to_csv(os.path.join(self.__path_to_folder, result_dataset_name))

    def get_correlation_matrix(self) -> pd.DataFrame:
        """
        Computes and returns the correlation matrix of the encoded dataset.

        Returns:
            pd.DataFrame: Correlation matrix of the encoded dataset.
        """
        corr_matrix = self.__encoded_dataset.corr()
        return corr_matrix


    def chi_squared_feature_selection(self) -> pd.DataFrame:
        """
        Performs Chi-Squared test for feature selection and returns the results in a sorted DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing features, Chi-Squared values, and p-values, sorted by Chi-Squared values.
        """
        X = self.__encoded_dataset
        y = self.__target_column
        chi2_results = []
        
        for column in X.columns:
            contingency_table = pd.crosstab(X[column], y)      
            # Accomplishing Chi-Squared Test
            chi2_stat, p_val, _, _ = chi2_contingency(contingency_table)
            chi2_results.append({
                'Feature': column,
                'Chi-Squared': chi2_stat,
                'p-Value': p_val
            })    
        results_df = pd.DataFrame(chi2_results).sort_values(by='Chi-Squared', ascending=False)        
        return results_df

    def mutual_info_feature_selection(self, mi_random_state: int) -> pd.DataFrame:
        """
        Selects features based on mutual information and returns the results in a sorted DataFrame.

        Args:
            mi_random_state (int): Random state for reproducibility in mutual information calculation.

        Returns:
            pd.DataFrame: A DataFrame containing features and their mutual information scores, sorted by score.
        """
        X = self.__encoded_dataset
        y = self.__target_column
        mutual_info_scores = mutual_info_classif(X, y, random_state = mi_random_state)
        mutual_info_result = pd.DataFrame({'Feature': X.columns, 'Score': mutual_info_scores})
        mutual_info_result = mutual_info_result.sort_values(by='Score', ascending=False)
        return mutual_info_result

    def random_forest_feature_selection(self, with_permutations: bool, permutation_repeats: int,
                                        permutation_random_state: int) -> pd.DataFrame:
        """
        Selects features using a Random Forest classifier. Optionally, permutation importance can be used.
        Returns the feature importances in a sorted DataFrame.

        Args:
            with_permutations (bool): Whether to use permutation importance for feature selection.
            permutation_repeats (int): Number of permutation repeats to calculate feature importance.
            permutation_random_state (int): Random state for reproducibility in permutation importance.

        Returns:
            pd.DataFrame: A DataFrame containing features and their importance scores, sorted by importance.
        """
        X = self.__encoded_dataset
        y = self.__target_column

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        rf = RandomForestClassifier(n_estimators=100, random_state=0)

        # To avoid overfitting we set up minimal sample number for a leaf
        rf.set_params(min_samples_leaf=20).fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Random Forest train accuracy: {rf.score(X_train, y_train):.2f}")
        print(f"Random Forest test accuracy: {accuracy:.2f}\n")
        
        if with_permutations:
            permutation_result = permutation_importance(rf, X_test, y_test, 
                                                        n_repeats=permutation_repeats, 
                                                        random_state=permutation_random_state)
            feature_importances = pd.DataFrame({
                'Feature': X.columns,
                'Importance': permutation_result.importances_mean,
                'Std': permutation_result.importances_std
            }).sort_values(by='Importance', ascending=False)
        else:
            importances = rf.feature_importances_
            feature_names = X.columns

            feature_importances = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)
        return feature_importances
    
    def get_mean_ranks_for_interpretation_results(self, sorted_chi2_result: pd.DataFrame,
                                                  sorted_mutual_info_result: pd.DataFrame, 
                                                  sorted_random_forest_result: pd.DataFrame) -> pd.DataFrame:
        """
        Computes and returns the mean rank of features based on their ranking from Chi-Squared,
        Mutual Information, and Random Forest feature selection methods.

        Args:
            sorted_chi2_result (pd.DataFrame): DataFrame containing features ranked by Chi-Squared values.
            sorted_mutual_info_result (pd.DataFrame): DataFrame containing features ranked by Mutual Information scores.
            sorted_random_forest_result (pd.DataFrame): DataFrame containing features ranked by Random Forest importance.

        Returns:
            pd.DataFrame: A DataFrame with features and their ranks from each method, along with the mean rank.
        """
        sorted_chi2_features = sorted_chi2_result["Feature"].tolist()
        sorted_mutual_info_features = sorted_mutual_info_result["Feature"].tolist()
        sorted_random_forest_features = sorted_random_forest_result["Feature"].tolist()
        
        features = self.__encoded_dataset.columns
        ranks = pd.DataFrame(index=features)
        ranks['chi_squared_rank'] = [sorted_chi2_features.index(feature) + 1 for feature in features]
        ranks['mutual_info_rank'] = [sorted_mutual_info_features.index(feature) + 1 for feature in features]
        ranks['random_forest_rank'] = [sorted_random_forest_features.index(feature) + 1 for feature in features]
        ranks['mean_rank'] = ranks.mean(axis=1)
        return ranks
