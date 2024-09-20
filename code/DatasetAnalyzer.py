import pandas as pd
import re

class DatasetAnalyzer:
    def __init__(self, dataset: pd.DataFrame):
        self.__analyzed_dataset = dataset
        self.__categorical_dataset = pd.DataFrame()
        self.__text_dataset = pd.DataFrame()

    def check_missing_values(self, percent_threshold: int):
        missing_val = round((self.__analyzed_dataset.isnull().sum())/(len(self.__analyzed_dataset))*100, 2)
        self.__missing_data_more_treshhold = pd.DataFrame(missing_val[missing_val.values>=percent_threshold],columns=['Missing percent'])
        self.__missing_data_less_treshhold = pd.DataFrame(missing_val[missing_val.values<percent_threshold],columns=['Missing percent'])

        print(f"Columns with more than {percent_threshold}% missed data: \n{self.__missing_data_more_treshhold}")
        print("\n")
        print(f"Columns with less than {percent_threshold}% missed data: \n{self.__missing_data_less_treshhold}")
    
    def drop_sparse_columns(self):
        self.__analyzed_dataset = self.__analyzed_dataset.drop(columns = self.__missing_data_more_treshhold.index)



    def preprocess_columns(self) -> None:
        self.__preprocess_age_and_gender()
        self.__divide_dataset_to_categorical_and_text()
        self.__fill_missing_numbers_less_threshold()
        self.__print_additional_info()

    def __preprocess_age_and_gender(self) -> None:
        for ith_column in self.__analyzed_dataset:
            if self.__analyzed_dataset[ith_column].dtype == 'object' or self.__analyzed_dataset[ith_column].dtype == 'string':
                self.__analyzed_dataset[ith_column] = self.__analyzed_dataset[ith_column].str.lower()
                # self.__analyzed_dataset[ith_column] = self.__analyzed_dataset[ith_column].str.strip(" ")
        
        # Replace similar gender names with only one.  
        self.__analyzed_dataset["What is your gender?"] = self.__analyzed_dataset["What is your gender?"].replace({
            'm': 'male',
            'man': 'male',
            'f': 'female',
            'woman': 'female'
        })
        # All other names will be replaced with 'trans'
        self.__analyzed_dataset["What is your gender?"] = self.__analyzed_dataset["What is your gender?"].\
            apply(lambda x: x if x in ['male', 'female'] else 'transgender')    
        # Now correct age column a little
        median_age = self.__analyzed_dataset["What is your age?"][(self.__analyzed_dataset["What is your age?"]>=18) & 
                                                          (self.__analyzed_dataset["What is your age?"]<=100)].median()
        self.__analyzed_dataset["What is your age?"] = self.__analyzed_dataset["What is your age?"].\
            apply(lambda x: x if 18 <= x <= 100 else median_age)

        def age_to_range(age):
            if 18 <= age <= 29:
                return '18-29'
            elif 30 <= age <= 39:
                return '30-39'
            elif 40 <= age <= 49:
                return '40-49'
            elif 50 <= age <= 59:
                return '50-59'
            elif 60 <= age <= 100:
                return '60-100'        
        self.__analyzed_dataset["What is your age?"] = self.__analyzed_dataset["What is your age?"].apply(age_to_range)

    def __divide_dataset_to_categorical_and_text(self) -> None: 
        categorical_columns = ["What is your age?", "What is your gender?", 
                               "What country do you live in?", "What country do you work in?",
                               "What US state or territory do you live in?", 
                               "What US state or territory do you work in?"]
        for i in range(self.__analyzed_dataset.shape[1]):
            ith_column = self.__analyzed_dataset.iloc[:, i]
            counts = ith_column.unique()
            if len(counts)>10 and ith_column.name not in categorical_columns:
                self.__text_dataset[ith_column.name] = ith_column
            else:
                self.__categorical_dataset[ith_column.name] = ith_column

    def __remove_parentheses(self, text: str) -> None:
        return re.sub(r'\([^)]*\)', '', text).replace("  ", " ").strip()

    def __fill_missing_numbers_less_threshold(self):        
        for ith_column in self.__text_dataset:
            modes = self.__text_dataset[ith_column].mode().iloc[0]
            self.__text_dataset[ith_column] = self.__text_dataset[ith_column].fillna(modes)    
            self.__text_dataset[ith_column] = self.__text_dataset[ith_column].apply(self.__remove_parentheses)
            self.__text_dataset[ith_column] = self.__text_dataset[ith_column].str.replace(r'\s+\|', '|', regex=True)

        # Here, we replace only values for columns where less than 75% of data is missed
        for ith_column in self.__categorical_dataset:
            # Compute the mode
            mode_value = self.__categorical_dataset[ith_column].mode()[0]
            self.__categorical_dataset.loc[self.__categorical_dataset[ith_column].isna(), ith_column] = mode_value    



    def __print_additional_info(self) -> None:
        print("\nAdditional information about prepared dataset:")
        print(f"Size of dataset that consists of categorical columns: {self.__categorical_dataset.shape}")
        print(f"Size of dataset that consists of text columns: {self.__text_dataset.shape}")
        print(f"Number of NaN values after filling: {self.__categorical_dataset.isnull().sum().sum() + self.__text_dataset.isnull().sum().sum()}\n")

    def return_divided_datasets(self) -> list:
        return self.__categorical_dataset, self.__text_dataset