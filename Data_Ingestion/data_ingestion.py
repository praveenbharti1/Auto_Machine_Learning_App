import pandas as pd
import numpy as np
from Logger import CustomLogger 

class FileIngestion:

    def __init__(self, file_path, file_format='csv', **kwargs):
        self.file_path = file_path
        self.file_format = file_format
        self.log = CustomLogger("log.log")

    def file(self):
        try:
            self.filepath_raw = "Raw_data/raw_file.csv"
            self.filepath_processed = "Pre_processed_data/processed_data.csv"

            
            data_raw = pd.read_csv(self.filepath_raw)
            data_processed = pd.read_csv(self.filepath_processed)
            self.log.log_info("Raw Data And Processed Data is saved to their respective folder")
            return data_raw, data_processed
        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")



    def read_file(self):
        """
        Read data from the specified file or data object.

        Returns:
            DataFrame or object: The ingested data.

        Example:
            After initializing the FileIngestion class, use the read_file method to load the data:

            >>> file_ingestion = FileIngestion('data.csv')
            >>> data = file_ingestion.read_file()

            For custom data types (not 'csv'), you can pass data directly to the class:

            >>> sample_data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
            >>> file_ingestion = FileIngestion(sample_data, file_format='custom')
            >>> data = file_ingestion.read_file()
        """
        try:
            if self.file_format == 'csv':
                if self.file_path:
                    data = pd.read_csv(self.file_path)
                    return data
                else:
                    self.log.log_info('ValueError("No file path provided for csv format.")')
                    raise ValueError("No file path provided for 'csv' format.") 
                    
            else:
                # Handle other file formats or custom data types here
                if self.custom_data is not None:
                    return self.custom_data
                else:
                    self.log.log_info('ValueError("No data provided for the specified format.")')
                    raise ValueError("No data provided for the specified format.")
        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")
                

    def drop_columns_with_high_missing(self,  data, missing_threshold=0.7):
        """
        Drop columns in a DataFrame with missing values exceeding a specified threshold.

        Parameters:
            data (DataFrame): The input DataFrame containing the data.
            missing_threshold (float): The threshold for the percentage of missing values in a column.
                Columns with missing percentages greater than this threshold will be dropped.

        Returns:
            DataFrame: The DataFrame with columns dropped based on the specified threshold.

        Example:
            Suppose you have a DataFrame 'my_data' with the following structure:

            >>> my_data
            A    B    C    D
            0  1.0  NaN  3.0  NaN
            1  2.0  2.0  NaN  4.0
            2  3.0  3.0  NaN  NaN
            3  4.0  NaN  NaN  4.0

            You can use the 'drop_columns_with_high_missing' function to remove columns with more than 70% missing values as follows:

            >>> cleaned_data = drop_columns_with_high_missing(my_data, missing_threshold=0.7)
            Dropping columns with more than 70% missing values: ['B', 'C']
            >>> cleaned_data
            A    D
            0  1.0  NaN
            1  2.0  4.0
            2  3.0  NaN
            3  4.0  4.0

            In this example, columns 'B' and 'C' have missing percentages greater than 70%, so they are dropped from the DataFrame.
        """
        try:
            # Calculate the percentage of missing values in each column
            missing_percentages = data.isnull().mean()

            # Identify columns with missing percentages greater than the threshold
            columns_to_drop = missing_percentages[missing_percentages > missing_threshold].index.tolist()

            if columns_to_drop:
                self.log.log_info(f"Dropping columns with more than {int(missing_threshold * 100)}% missing values: {columns_to_drop}")
                data.drop(columns=columns_to_drop, inplace=True)
            else:
                self.log.log_info("No columns found with more than the specified threshold of missing values.")
            return data
        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")



    def remove_duplicated_columns(self, data):
        """
        Identify and remove duplicated columns in a DataFrame.

        Parameters:
            data (DataFrame): The input DataFrame containing the data.

        Returns:
            None

        Example:
            Remove duplicated columns from a DataFrame, if any.

            Suppose you have a DataFrame 'my_data' with the following structure:

            >>> my_data
            A    B    C    A
            0  1.0  2.0  3.0  1.0
            1  2.0  2.0  4.0  2.0
            2  3.0  3.0  5.0  3.0
            3  4.0  4.0  6.0  4.0

            You can use the 'remove_duplicated_columns' method to remove duplicated columns as follows:

            >>> file_ingestion = FileIngestion('my_data.csv')
            file_ingestion.remove_duplicated_columns(file_ingestion.data)
            Dropping duplicated columns: ['A']
            >>> file_ingestion.data
            B    C
            0  2.0  3.0
            1  2.0  4.0
            2  3.0  5.0
            3  4.0  6.0

            In this example, the column 'A' is duplicated, and it is removed from the DataFrame, leaving only one instance of it.
        """
        try:
            # Identify and remove duplicated columns
            duplicated_columns = data.columns[data.columns.duplicated()]

            if len(duplicated_columns) > 0:
                self.log.log_info(f"Dropping duplicated columns: {duplicated_columns.tolist()}")
                data.drop(columns=duplicated_columns, inplace=True)
                return data
            else:
                self.log.log_info("No duplicated columns found in the dataset.")
        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")


    
    def remove_unnamed_column(self, data):
        """
        Remove the 'Unnamed: 0' column from the DataFrame if it exists.

        Args:
            data (pd.DataFrame): The DataFrame to process.

        Returns:
            pd.DataFrame: The DataFrame with the 'Unnamed: 0' column removed.
        """
        if 'Unnamed: 0' in data.columns:
            data = data.drop(columns=['Unnamed: 0'])
            self.log.log_info("dropped unnamed columns")
        return data
    

    def drop_columns_with_date_format(self, data):
        """
        Drop columns containing data in the format 'dd-mm-yy'.

        Args:
            data (pd.DataFrame): The DataFrame to process.

        Returns:
            pd.DataFrame: The DataFrame with columns removed.
        """
        try:
            # Initialize a list to store columns to drop
            columns_to_drop = []

            # Iterate over columns and check if all values match the date format 'dd-mm-yy'
            for column in data.columns:
                is_date_format = all(pd.to_datetime(data[column], errors='coerce').notna())
                if is_date_format:
                    columns_to_drop.append(column)

            # Drop the identified columns
            data = data.drop(columns=columns_to_drop)
            self.log.log_info("Dropping the colummns with date and time")

            return data
        
        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")


    def rejecting_file(self):
        pass
