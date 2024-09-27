import pandas as pd

class WorkingDatasetInfo:
    """
    A class for providing information about a working dataset.

    Attributes:
        __dataset (pd.DataFrame): The dataset to work with.
    """
    
    def __init__(self, dataset: pd.DataFrame):
        """
        Initializes the WorkingDatasetInfo object with a dataset.

        Args:
            dataset (pd.DataFrame): The dataset to work with.
        """
        self.__dataset = dataset

    def print_dataset_info(self) -> None:
        """
        Prints information about the dataset, including its size and the count of each column type.
        """
        self.__print_dataset_size()
        self.__print_columns_type_count()

    def __print_dataset_size(self) -> None:
        """
        Prints the size of the dataset (number of rows and columns).
        """
        print(f"Size of original dataset: {self.__dataset.shape}")

    def __print_columns_type_count(self) -> None:
        """
        Prints the count of each column type in the dataset.
        """
        column_types_count = self.__dataset.dtypes.value_counts()
        print(column_types_count)

    def print_each_column_types(self) -> None:
        """
        Prints the data type of each column in the dataset.
        """
        for ith_column in self.__dataset:
            print(f"Type of the column \'{ith_column}\':\n{self.__dataset[ith_column].dtypes}")

    def get_dataset_columns(self) -> pd.Index:
        """
        Returns the columns of the dataset.

        Returns:
            pd.Index: The columns of the dataset.
        """
        return self.__dataset.columns
