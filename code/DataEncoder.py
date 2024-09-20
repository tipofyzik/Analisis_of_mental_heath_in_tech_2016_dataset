from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class DataEncoder:
    def __init__(self) -> None:
        self.__label_encoders = {}

        self.__labeled_columns = []
        self.__binary_columns = []
        self.__specific_columns = [
            "What country do you live in?", 
            "What country do you work in?",
            "What US state or territory do you live in?", 
            "What US state or territory do you work in?"]
        
    def pass_text_columns(self, text_columns: list[str]):
        self.__labeled_columns.extend(text_columns)

    def __define_binary_columns(self, dataset: pd.DataFrame) -> None:
        for ith_column in dataset:
            unique_values = dataset[ith_column].unique()
            if len(unique_values) == 2:
                self.__binary_columns.append(ith_column)

    def __define_columns_to_label(self, dataset: pd.DataFrame) -> None:
        self.__define_binary_columns(dataset)
        self.__labeled_columns.extend(self.__binary_columns)
        self.__labeled_columns.extend(self.__specific_columns)

    def encode_data(self, dataset: pd.DataFrame):
        self.__define_columns_to_label(dataset)

        #Label encoding for binary and specific columns
        labeled_dataset = dataset[self.__labeled_columns].copy()
        for column in labeled_dataset.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            labeled_dataset.loc[:, column] = le.fit_transform(labeled_dataset[column])
            self.__label_encoders[column] = le

        #One-hot encoding for other columns
        one_hot_dataset = pd.get_dummies(dataset.drop(columns = self.__labeled_columns), dtype=np.uint8)
        self.__encoded_dataset = pd.concat([one_hot_dataset, labeled_dataset], axis=1)

    def normalize_data(self):
        scaler = StandardScaler()
        self.__normalized_data = scaler.fit_transform(self.__encoded_dataset)

    def get_encoded_dataset(self):
        print(f"\nSize of encoded dataset: {self.__normalized_data.shape}\n")
        return self.__encoded_dataset, self.__normalized_data