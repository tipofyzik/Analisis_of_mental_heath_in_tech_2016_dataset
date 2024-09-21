from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class DataEncoder:
    """
    A class to handle the encoding of categorical and textual data within a dataset,
    including label encoding for specific columns and one-hot encoding for others.

    Attributes:
        __label_encoders: A dictionary to store the label encoders for each column.
        __labeled_columns: A list of columns to be label encoded.
        __binary_columns: A list of columns identified as binary.
        __specific_columns: A predefined list of specific columns to be included in label encoding.
    """
    
    def __init__(self):
        """
        Initializes the DataEncoder with empty label encoders and column lists.
        """
        self.__label_encoders = {}

        self.__labeled_columns = []
        self.__binary_columns = []
        self.__specific_columns = [
            "What country do you live in?", 
            "What country do you work in?",
            "What US state or territory do you live in?", 
            "What US state or territory do you work in?"]
        
    def pass_text_columns(self, text_columns: list[str]) -> None:
        """
        Accepts a list of text columns to be included for label encoding.

        Args:
            text_columns (list[str]): A list of text column names.
        """
        self.__labeled_columns.extend(text_columns)

    def __define_binary_columns(self, dataset: pd.DataFrame) -> None:
        """
        Identifies and stores binary columns from the dataset based on unique values.

        Args:
            dataset (pd.DataFrame): The dataset from which to identify binary columns.
        """
        for ith_column in dataset:
            unique_values = dataset[ith_column].unique()
            if len(unique_values) == 2:
                self.__binary_columns.append(ith_column)

    def __define_columns_to_label(self, dataset: pd.DataFrame) -> None:
        """
        Defines which columns should be label encoded by combining binary and specific columns.

        Args:
            dataset (pd.DataFrame): The dataset from which to define columns for label encoding.
        """
        self.__define_binary_columns(dataset)
        self.__labeled_columns.extend(self.__binary_columns)
        self.__labeled_columns.extend(self.__specific_columns)

    def encode_data(self, dataset: pd.DataFrame) -> None:
        """
        Encodes the dataset by applying label encoding to specified columns and 
        one-hot encoding to other categorical columns.

        Args:
            dataset (pd.DataFrame): The dataset to be encoded.
        """
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

    def normalize_data(self) -> None:
        """
        Normalizes the encoded dataset using standard scaling.
        """
        scaler = StandardScaler()
        self.__normalized_data = scaler.fit_transform(self.__encoded_dataset)

    def get_encoded_dataset(self) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Returns the encoded dataset and the normalized data, while also printing 
        the size of the encoded dataset.

        Returns:
            tuple[pd.DataFrame, np.ndarray]: A tuple containing the encoded dataset and the normalized data.
        """
        print(f"\nSize of encoded dataset: {self.__normalized_data.shape}\n")
        return self.__encoded_dataset, self.__normalized_data