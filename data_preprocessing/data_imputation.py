import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from Logger import CustomLogger
import joblib
import os


class DataProcessing:

    def __init__(self, null_threshold=0.03, k_neighbors=2, iqr_threshold=1.5):
        """
        Initialize the DataProcessing class with optional configuration parameters.

        Parameters:
            null_threshold (float, optional): The threshold for considering whether to drop rows with missing values.
                                             If the percentage of missing values is less than this threshold, rows are dropped.
                                             Defaults to 0.03.
            k_neighbors (int, optional): The number of neighbors to use for KNN imputation.
                                        Defaults to 2.
            iqr_threshold (float, optional): The threshold for identifying outliers using the IQR method.
                                            Defaults to 1.5.
        """
        self.null_threshold = null_threshold
        self.k_neighbors = k_neighbors
        self.iqr_threshold = iqr_threshold
        self.log = CustomLogger("log.log")

    def drop_duplicates(self, data):
        """
        Remove duplicate rows from the DataFrame, if any.

        Parameters:
            data (pd.DataFrame): The DataFrame to check for duplicates and remove them if found.

        Returns:
            None. The method operates in-place and modifies the input DataFrame.

        Examples:
        --------
        >>> dp = DataProcessing()
        >>> data = pd.DataFrame({'A': [1, 2, 2, 3, 4], 'B': [5, 6, 6, 7, 8]})
        >>> dp.drop_duplicates(data)
        Duplicates found in the DataFrame.
        Duplicates have been dropped.

        >>> data
           A  B
        0  1  5
        1  2  6
        3  3  7
        4  4  8

        >>> data = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]})
        >>> dp.drop_duplicates(data)
        No duplicates found in the DataFrame.

        >>> data
           A  B
        0  1  5
        1  2  6
        2  3  7
        3  4  8
        """
        try:
            if data.duplicated().any():
                self.log.log_info("Duplicates found in the DataFrame.")
                # Drop duplicates
                data.drop_duplicates(inplace=True)
                self.log.log_info("Duplicates have been dropped.")
            else:
                self.log.log_info("No duplicates found in the DataFrame.")
        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")


    def drop_constant_columns(self, data):
        """
        Remove columns with constant values (all values are the same) from the DataFrame.

        Parameters:
            data (pd.DataFrame): The DataFrame to check for constant columns and remove them if found.

        Returns:
            None. The method operates in-place and modifies the input DataFrame.

        Examples:
        --------
        >>> dp = DataProcessing()
        >>> data = pd.DataFrame({'A': [1, 1, 1, 1], 'B': [2, 2, 2, 2]})
        >>> dp.drop_constant_columns(data)
        Constant columns found in the DataFrame. Dropping them.
        Constant columns have been dropped.

        >>> data
           B
        0  2
        1  2
        2  2
        3  2

        >>> data = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [2, 2, 2, 2]})
        >>> dp.drop_constant_columns(data)
        No constant columns found in the DataFrame.

        >>> data
           A  B
        0  1  2
        1  2  2
        2  3  2
        3  4  2
        """
        try:
            constant_columns = data.columns[data.nunique() == 1]
            if constant_columns.any():
                self.log.log_info(
                    "Constant columns found in the DataFrame. Dropping them.")
                data.drop(columns=constant_columns, inplace=True)
                self.log.log_info("Constant columns have been dropped.")
            else:
                self.log.log_info("No constant columns found in the DataFrame.")
        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")


    def impute_outliers_with_iqr(self, data):
        """
        Impute outliers in the DataFrame using the Interquartile Range (IQR) method.

        Parameters:
            data (pd.DataFrame): The DataFrame to detect and impute outliers in.

        Returns:
            None. The method operates in-place and modifies the input DataFrame.

        Examples:
        --------
        >>> dp = DataProcessing()
        >>> data = pd.DataFrame({'A': [1, 2, 3, 4, 5, 100], 'B': [5, 6, 7, 8, 9, 200]})
        >>> dp.impute_outliers_with_iqr(data)
        Outliers found in column 'A'. Imputing with IQR method.
        Outliers found in column 'B'. Imputing with IQR method.

        >>> data
           A  B
        0  1  5
        1  2  6
        2  3  7
        3  4  8
        4  5  9
        5  8  8

        >>> data = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [5, 6, 7, 8, 9]})
        >>> dp.impute_outliers_with_iqr(data)
        No outliers found in the DataFrame.

        >>> data
           A  B
        0  1  5
        1  2  6
        2  3  7
        3  4  8
        4  5  9
        """
        try:
            # Define a threshold for identifying outliers
            iqr_threshold = 1.5

            # Check for outliers in each column
            for column in data.columns:
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - iqr_threshold * IQR
                upper_bound = Q3 + iqr_threshold * IQR

                outliers = data[(data[column] < lower_bound) |
                                (data[column] > upper_bound)]

                if not outliers.empty:
                    self.log.log_info(
                        f"Outliers found in column '{column}'. Imputing with IQR method.")
                    data[column] = data[column].clip(
                        lower=lower_bound, upper=upper_bound)
                else:
                    self.log.log_info(f"No outliers found in column '{column}'.")
        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")


    def handling_missing_values(self, data, null_threshold=0.03, k_neighbors=2):
        """
        Handle missing values in the DataFrame by either dropping rows or imputing with K-nearest neighbors (KNN).

        Parameters:
            data (pd.DataFrame): The DataFrame with missing values.
            null_threshold (float, optional): The threshold for considering whether to drop rows with missing values.
                                             If the percentage of missing values is less than this threshold, rows are dropped.
                                             Defaults to 0.03.
            k_neighbors (int, optional): The number of neighbors to use for KNN imputation.
                                        Defaults to 2.

        Returns:
            pd.DataFrame: The DataFrame with missing values handled (either rows dropped or imputed using KNN).

        Examples:
        --------
        >>> dp = DataProcessing()
        >>> data = pd.DataFrame({'A': [1, 2, None, 4, 5], 'B': [5, None, 7, None, 9]})
        >>> dp.handling_missing_values(data)
        Null values are equal to or greater than the specified threshold. Imputing with KNN.

        >>> data
             A    B
        0  1.0  5.0
        1  2.0  6.0
        2  2.25 7.0
        3  4.0  7.25
        4  5.0  9.0

        >>> data = pd.DataFrame({'A': [1, 2, None, 4, 5], 'B': [5, None, 7, None, 9]})
        >>> dp.handling_missing_values(data, null_threshold=0.5)
        Null values are less than the specified threshold. Dropping rows with null values.

        >>> data
             A    B
        0  1.0  5.0
        3  4.0  NaN
        4  5.0  9.0
        """
        try:
            # Calculate the percentage of null values in the DataFrame
            null_percentage = data.isnull().sum().sum() / \
                (data.shape[0] * data.shape[1])

            # Check if null values are less than the threshold
            if null_percentage < null_threshold:
                self.log.log_info(
                    "Null values are less than the specified threshold. Dropping rows with null values.")
                data.dropna(inplace=True)
            else:
                self.log.log_info(
                    "Null values are equal to or greater than the specified threshold. Imputing with KNN.")
                # Create a KNNImputer with the desired number of neighbors
                imputer = KNNImputer(n_neighbors=k_neighbors)
                data_imputed = imputer.fit_transform(data)
                data = pd.DataFrame(data_imputed, columns=data.columns)

            return data
        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")


    def impute_missing_values_with_knn(self, data, k_neighbors=2):
        """
        Impute missing values in the DataFrame using K-nearest neighbors (KNN) imputation.

        Parameters:
            data (pd.DataFrame): The DataFrame with potentially missing values to impute.
            k_neighbors (int, optional): The number of neighbors to use for imputation. Defaults to 2.

        Returns:
            pd.DataFrame: The DataFrame with imputed values using K-nearest neighbors.

        Examples:
        --------
        >>> dp = DataProcessing()
        >>> data = pd.DataFrame({'A': [1, 2, 3, 4, None], 'B': [5, 6, None, 8, 9]})
        >>> imputed_data = dp.impute_missing_values_with_knn(data)
        Missing values have been imputed using K-nearest neighbors.

        >>> imputed_data
            A     B
        0  1.0  5.00
        1  2.0  6.00
        2  3.0  6.25
        3  4.0  8.00
        4  2.5  9.00
        """
        try:
            # Create a KNNImputer with the desired number of neighbors
            imputer = KNNImputer(n_neighbors=k_neighbors)

            # Perform imputation
            data_imputed = imputer.fit_transform(data)

            # Convert the result back to a DataFrame
            data_imputed = pd.DataFrame(data_imputed, columns=data.columns)
            self.log.log_info(
                "Missing values have been imputed using K-nearest neighbors.")

            return data_imputed
        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")


    def encode_categorical_columns(self, data):
        """
        Encode categorical columns using LabelEncoder.

        Parameters:
            data (pd.DataFrame): The DataFrame to encode categorical columns in.

        Returns:
            pd.DataFrame: The DataFrame with categorical columns encoded.

        Example:
        --------
        >>> dp = DataProcessing()
        >>> data = pd.DataFrame({'Category': ['A', 'B', 'A', 'C'], 'Value': [1, 2, 3, 4]})
        >>> encoded_data = dp.encode_categorical_columns(data)
        Categorical columns have been encoded using LabelEncoder.

        >>> encoded_data
           Category  Value
        0         0      1
        1         1      2
        2         0      3
        3         2      4
        """
        try:
            # Create a dictionary to store the label encoder models
            label_encoders = {}

            # Loop through the categorical columns and fit label encoders
            for column in data.select_dtypes(include=['object']).columns:
                label_encoder = LabelEncoder()
                label_encoder.fit(data[column])
                label_encoders[column] = label_encoder

                if not os.path.exists("Model"):
                    os.makedirs("Model")

            joblib.dump(label_encoders, r"Model/label_encode.joblib")
            encoded_df = data.copy()
            for column, label_encoder in label_encoders.items():
                encoded_df[column] = label_encoder.transform(data[column])

            encoded_df = data.copy()
            for column, label_encoder in label_encoders.items():
                encoded_df[column] = label_encoder.transform(data[column])

            self.log.log_info("Categorical columns have been encoded using LabelEncoder.")
            return encoded_df
        except Exception as e:
            self.log.log_info(f"this is error and your error is :: {str(e)} ")


    def drop_columns_with_same_index_values(self, data):
        """
        Drop columns with the same values as the index.

        Args:
            data (pd.DataFrame): The DataFrame to process.

        Returns:
            pd.DataFrame: The DataFrame with columns removed.
        """
        try:
            # Get the index values
            index_values = data.index.values

            # Iterate over columns and check if all values in a column match the index
            columns_to_drop = [col for col in data.columns if all(
                data[col] == index_values)]

            if columns_to_drop:
                self.log.log_info(f"Dropping columns with the same values as the index: {columns_to_drop}")
                # Drop the identified columns
                data = data.drop(columns=columns_to_drop)
            else:
                self.log.log_info("No columns found with the same values as the index.")
            return data
        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")


    def drop_columns_with_counting_numbers(self, data):
        """
        Drop columns containing counting numbers (sequential numbers).

        Args:
            data (pd.DataFrame): The DataFrame to process.

        Returns:
            pd.DataFrame: The DataFrame with columns removed.
        """
        try:
            # Initialize a list to store columns to drop
            columns_to_drop = []

            # Iterate over columns and check if all values are counting numbers
            for column in data.columns:
                is_counting = all(data[column] == list(range(1, len(data) + 1)))
                if is_counting:
                    columns_to_drop.append(column)

            if columns_to_drop:
                self.log.log_info(f"Dropping columns containing counting numbers: {columns_to_drop}")
                # Drop the identified columns
                data = data.drop(columns=columns_to_drop)
            else:
                self.log.log_info("No columns found containing counting numbers.")

            return data
        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")

    
    

    def check_counting_same_index_columns(self, data):
        try:
            # Make a copy of the original data for comparison
            original_data = data.copy()
            
            label_encoder = LabelEncoder()
            for column in data.select_dtypes(include=['object']).columns:
                data[column] = label_encoder.fit_transform(data[column])

        

            # Create a list to store columns to drop
            columns_to_drop = []

            # Iterate over columns and check if all values are counting numbers
            for column in data.columns:
                index_values = data.index.values

                is_counting = (all(data[column] == list(range(1, len(data) + 1)))) or (all(data[column] == index_values)) or (all(data[column] == list(range(0, len(data)))))
                if is_counting:
                    columns_to_drop.append(column)
                else:
                    self.log.log_info("No columns found containing counting numbers.")
                
            # Return the original data with only the specified columns dropped
            original_data = original_data.drop(columns=columns_to_drop, axis=1)
            

            # Create the "categorical_data" folder if it doesn't exist
            if not os.path.exists("Modified_Data"):
                os.makedirs("Modified_Data")
            
            # Save the modified data as a CSV file in the "categorical_data" folder
            original_data.to_csv('Modified_Data/modified_data.csv', index=False)
            
            self.log.log_info("Modified data saved as 'modified_data.csv' in the 'categorical_data' folder.")
                
            return original_data
        except Exception as e:
            self.log.log_info(f"this is error and your error is :: {str(e)} ")

        
            



