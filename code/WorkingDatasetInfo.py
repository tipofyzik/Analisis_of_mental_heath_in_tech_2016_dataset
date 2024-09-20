import pandas as pd

class WorkingDatasetInfo:
    def __init__(self, dataset: pd.DataFrame):
        self.__dataset = dataset

    def print_dataset_info(self):
        self.__print_dataset_size()
        self.__print_columns_type_count()

    def __print_dataset_size(self):
        print(f"Size of original dataset: {self.__dataset.shape}")

    def __print_columns_type_count(self):
        column_types_count = self.__dataset.dtypes.value_counts()
        print(column_types_count)

    def print_each_column_types(self):
        for ith_column in self.__dataset:
            print(f"Type of the column \'{ith_column}\':\n{self.__dataset[ith_column].dtypes}")

    def get_dataset_columns(self):
        return self.__dataset.columns
