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

    def save_unique_values_with_counts_to_csv_table(self) -> None:
        """
        Creates and saves a table with unique values and their counts for each column in the dataset.
        
        The table consists of:
        - A 'Questions' column listing the dataset's column names.
        - A 'Number of unique values' column showing the count of unique values per column.
        - Additional columns showing unique values and their respective frequencies, with NaN values represented as 'NaN'.
        
        The resulting table is saved as 'analysis_result.csv' in the current working directory.
        """
        result_data = []
        header_row = ['Questions', 'Number of unique values']
        for column in self.__dataset.columns:
            unique_values_with_counts = self.__dataset[column].value_counts(dropna=False)
            unique_values_with_counts.index = unique_values_with_counts.index.fillna('NaN')
            unique_count = unique_values_with_counts.size

            row = [column, unique_count] + [f"{value}: {count}" for value, count in unique_values_with_counts.items()]
            result_data.append(row)

        max_length = max(len(row) for row in result_data)
        for row in result_data:
            row.extend([""] * (max_length - len(row)))
        header_row.extend([f'Unique value {i}' for i in range(1, max_length - 1)])
        result_data.insert(0, header_row)
        result_df = pd.DataFrame(result_data)

        file_name = "./results/analysis_result.csv"
        result_df.to_csv(file_name, index=False, header=False)

    def get_dataset_columns(self) -> pd.Index:
        """
        Returns the columns of the dataset.

        Returns:
            pd.Index: The columns of the dataset.
        """
        return self.__dataset.columns
