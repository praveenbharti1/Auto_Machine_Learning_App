import pandas as pd
from ydata_profiling import ProfileReport
import streamlit as st
import os
from streamlit_pandas_profiling import st_profile_report
from Logger import CustomLogger
import warnings
warnings.filterwarnings('ignore')


class Ex_Data_Ana:

    def __init__(self, filepath=""):
        self.filepath = filepath
        self.log = CustomLogger("log.log")

    def exploratory_data_analysis(self):
        try:
            # Created a title for the app
            self.log.log_info("this completed")
            # Created a styled container for the welcome message
            st.markdown("""
            <div style="background-color: #3498db; padding: 20px; border-radius: 10px; box-shadow: 5px 5px 5px #888888;">
                <h1 style="color: white; text-align: center;">Welcome to the Exploratory Data Analysis Process</h1>
                <p style="color: white; text-align: center; font-size: 18px;">This app allows you to perform EDA on your dataset.</p>
            </div>
            """, unsafe_allow_html=True)

            # More information about EDA
            st.header("What is EDA?")
            st.write("Exploratory Data Analysis (EDA) is a crucial step in data analysis. It involves the process of summarizing the main characteristics of a dataset, often with the help of visual methods. EDA helps in understanding the data, detecting anomalies, and making data-driven decisions.")

            st.header("How to Use this App")
            st.write("1. Upload your dataset in CSV format.")
            st.write("2. This app will generate an EDA report, including data statistics, visualizations, and more.")
            st.write("3. Explore the EDA report to gain insights into your data.")

            st.markdown('<div style="text-align: center"><h4> 📈 "Data Tales Unveiled: Your EDA Revelations 🧐" 📊 </div>', unsafe_allow_html=True)

            # Check if the file is not None
            if os.path.exists(self.filepath):
                # Read the file as a pandas DataFrame
                df = pd.read_csv(self.filepath)

                if 'Unnamed: 0' in df.columns:
                    df = df.drop(columns=['Unnamed: 0'])

                # Create a profile report object
                pr = ProfileReport(df)

                st.write("""\n""")

                # Display the profile report
                if st.button("Click For Exporatory Data Analysis"):
                    st_profile_report(pr)

                self.log.log_info("Exploratory data Analysis completed")

            else:
                st.write(f"The file '{self.filepath}' does not exist.")
        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")





