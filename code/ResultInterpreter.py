from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import chi2_contingency

from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os

class ResultInterpreter:
    def __init__(self, dataset: pd.DataFrame, result_dataset_name: str,
                 main_folder: str, dataset_folder: str):
        self.__path_to_folder = os.path.join(main_folder, dataset_folder)
        self.__save_dataset_to_file(dataset, result_dataset_name)
        self.__encoded_dataset, self.__target_column = self.__preprocess_for_interpretation(dataset)

    def __preprocess_for_interpretation(self, dataset: pd.DataFrame):
        label_encoders = {}
        X = dataset.drop(columns=['Cluster'])
        y = dataset['Cluster']
        
        for column in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column])
            label_encoders[column] = le
        return X, y
    
    def __save_dataset_to_file(self, dataset: pd.DataFrame, result_dataset_name: str):
        os.makedirs(self.__path_to_folder, exist_ok=True)
        dataset.to_csv(os.path.join(self.__path_to_folder, result_dataset_name))

    def get_correlation_matrix(self):
        corr_matrix = self.__encoded_dataset.corr()
        return corr_matrix

    def chi_squared_feature_selection(self):
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

    def mutual_info_feature_selection(self, number_of_features: int):
        X = self.__encoded_dataset
        y = self.__target_column
        selector = SelectKBest(score_func=mutual_info_classif, k=number_of_features)
        X_selected = selector.fit_transform(X, y)
        
        selected_features = X.columns[selector.get_support()]
        scores = selector.scores_

        # Создание DataFrame для отображения результатов
        mutual_info_result = pd.DataFrame({'Feature': X.columns, 'Score': scores})
        mutual_info_result = mutual_info_result.sort_values(by='Score', ascending=False)
        return mutual_info_result

    def random_forest_feature_selection(self, with_permutations: bool):
        X = self.__encoded_dataset
        y = self.__target_column

        # Training before feature selection
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        rf = RandomForestClassifier(n_estimators=100, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        # Оценка точности модели
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Random Forest Accuracy: {accuracy:.2f}")
        
        if with_permutations:
            permutation_result = permutation_importance(rf, X_test, y_test, n_repeats=30, random_state=0)
            feature_importances = pd.DataFrame({
                'Feature': X.columns,
                'Importance': permutation_result.importances_mean,  # Используем важность из permutation
                'Std': permutation_result.importances_std  # Стандартное отклонение
            }).sort_values(by='Importance', ascending=False)
        else:
            #Feature selection
            importances = rf.feature_importances_
            feature_names = X.columns
            # Создание DataFrame для визуализации
            feature_importances = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)
        return feature_importances
