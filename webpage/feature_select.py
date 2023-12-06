import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Logger import CustomLogger
import os
import warnings
warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)


class FeatureSelection:
    def __init__(self):
        filepath_processed = "Pre_processed_data/processed_data.csv"
        processed_data = pd.read_csv(filepath_processed)
        self.processed_data = processed_data
        self.selected_column = None
        self.log = CustomLogger("log.log")

    def display(self):
        try:
            st.markdown('<div style="text-align: center"><h2>🔍Feature Selection📊</div>', unsafe_allow_html=True)
            st.markdown(
                '<div style="background-color: #3498db; padding: 20px; border-radius: 10px; box-shadow: 5px 5px 5px #888888;">'
                '<h1 style="color: white; text-align: center;">Welcome to Feature Selection Process</h1>'
                '<p style="color: white; text-align: center; font-size: 18px;">Choose the output column for your analysis.</p>'
                '</div>',
                unsafe_allow_html=True
            ) 
            st.write("")
            st.write("🔍 Unlock the Power of Feature Selection! 🚀")
            st.write("Feature selection is a pivotal step in data analysis. It's your opportunity to handpick the output column that holds the key to your analysis success. 🗝️")
            st.write("In this section, you can select the output column for your analysis and explore the correlation heatmap to make informed decisions about feature selection. Let's chart your course to data excellence!")
            
            st.write("")  # Add space

            # Get the column names from the DataFrame
            column_names = self.processed_data.columns

            # Set a maximum width for a single line of buttons
            max_buttons_per_row = 3  # Adjust this value as needed

            # Create a horizontal layout with spacing
            columns = st.columns(max_buttons_per_row)

            # Display a button for each column
            for i, column in enumerate(column_names):
                if i % max_buttons_per_row == 0:
                    # Create a new row after reaching the maximum number of buttons per row
                    columns = st.columns(max_buttons_per_row)

                with columns[i % max_buttons_per_row]:
                    if st.button(column):
                        self.selected_column = column
                        if not os.path.exists("Pre_Trained_Data"):
                            os.makedirs("Pre_Trained_Data")

                        data = pd.read_csv("Pre_processed_data/processed_data.csv")
                        data[self.selected_column].to_csv(
                            "Pre_Trained_Data/dependent.csv", index=False)
            # Add space
            st.write("")

            # Display the selected column name (if any)
            if self.selected_column:
                st.success(f"Selected column: {self.selected_column}")
            else:
                st.info("Select a column by clicking the buttons above.")

            st.write("")
            st.write("Feature selection is your secret sauce to crafting the perfect analysis. It's all about cherry-picking the most relevant columns, giving your data the VIP treatment it deserves. 🎩✨")
            st.write("This step isn't just about reducing dimensionality; it's the key to unleashing your model's full potential. Get ready to boost performance and discover the hidden gems in your dataset! 💎🚀")
        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")


    def get_X_y(self):
        try:
            if self.selected_column:
                X = self.processed_data.drop(columns=[self.selected_column])
                y = self.processed_data[self.selected_column]
                

                return X, y
            else:
                return None, None
        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")


    def display_correlation_heatmap(self):
        try:
            X, y = FeatureSelection.get_X_y(self)

            st.title("Correlation Heatmap")

            # Calculate the correlation matrix
            corr_matrix = pd.DataFrame(X).corr()

            # Create a heatmap of the correlation matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
            st.pyplot()

            # Find columns with correlations greater than 0.80
            high_corr_columns = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    if abs(corr_matrix.iloc[i, j]) > 0.80:
                        high_corr_columns.append(
                            (corr_matrix.columns[i], corr_matrix.columns[j]))

            if high_corr_columns:
                st.success("Columns with correlations greater than 0.80:")
                for cols in high_corr_columns:
                    st.write(f"{cols[0]} and {cols[1]}")

                # Allow the user to drop one of the columns with high correlation
                column_to_drop = st.selectbox("Select a column to drop:", [
                                            col[0] for col in high_corr_columns])

                if column_to_drop:
                    X = X.drop(columns=[column_to_drop])
                    st.success(f"{column_to_drop} has been dropped.")
                    return X
            else:
                st.info(
                    "No columns have higher correlations with each other. All columns have either moderate or low correlations.")
                return X
        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")



