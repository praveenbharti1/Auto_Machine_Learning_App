from Data_Ingestion.data_ingestion import FileIngestion
from data_preprocessing.data_imputation import DataProcessing
from Logger import CustomLogger
import pandas as pd
import os


class DataClean:
    def __init__(self, file_path):
        self.file_path = file_path
        self.file_ingestion = FileIngestion(file_path)
        self.dp = DataProcessing()
        self.log = CustomLogger("log.log")

    def clean_data(self):
        try:
            # Read the data
            data = self.file_ingestion.read_file()

            # Drop columns with high missing values
            data = self.file_ingestion.drop_columns_with_high_missing(data)

            self.file_ingestion.remove_unnamed_column(data)

            # Remove duplicated columns
            self.file_ingestion.remove_duplicated_columns(data)

            self.file_ingestion.drop_columns_with_date_format(data)

            # Step 1: Remove duplicate rows
            self.dp.drop_duplicates(data)

            # Step 2: Remove constant columns
            self.dp.drop_constant_columns(data)

            # Step 4: Handle missing values
            data = self.dp.handling_missing_values(data)

            data = self.dp.check_counting_same_index_columns(data)
           

            data = pd.read_csv(r"Modified_Data/modified_data.csv")
            # Step 6: Encode categorical columns
            data = self.dp.encode_categorical_columns(data)
            
            # # Step 5: Impute missing values with K-nearest neighbors
            data = self.dp.impute_missing_values_with_knn(data)
            
            # data = self.dp.drop_columns_with_same_index_values(data)

            # data = self.dp.drop_columns_with_counting_numbers(data)

            # Step 3: Impute Outliers with IQR Method
            # self.dp.impute_outliers_with_iqr(data)
            self.log.log_info("Data cleaning completed.")
            
            return data
        
        except Exception as e:
            self.log.log_info(f"this is error and your error is :: {str(e)} ")


    def save_cleaned_data(self):
        try:
            if not os.path.exists("Pre_processed_data"):
                os.makedirs("Pre_processed_data")
            output_file_path = "Pre_processed_data/processed_data.csv"
            cleaned_data = self.clean_data()
            cleaned_data.to_csv(output_file_path, index=False)
            self.log.log_info(
                "Cleaned data saved to Pre_processed_data/processed_data.csv.")
        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")

