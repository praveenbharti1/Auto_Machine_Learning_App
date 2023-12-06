from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import pandas as pd
import streamlit as st
from Logger import CustomLogger
from webpage.EDA import Ex_Data_Ana
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessing:
    def __init__(self):
        filepath = "Raw_data/raw_file.csv"
        raw_data = pd.read_csv(filepath)
        self.raw_data = raw_data

        filepath_processed = "Pre_processed_data/processed_data.csv"
        processed_data = pd.read_csv(filepath_processed)
        self.processed_data = processed_data
        self.log = CustomLogger("log.log")

    def process_data(self):
        try:
            st.markdown('<div style="text-align: center"><h2> 🌟Welcome to Data Preprocessing Method!</div>', unsafe_allow_html=True)
            st.markdown('Unleash the power of data preprocessing to transform your raw data into valuable insights. 🛠️ Clean, structure, and optimize your data for analysis and machine learning. Get started now! 🚀')
            st.markdown('<div style="background-color: #3498db; padding: 20px; border-radius: 10px; box-shadow: 5px 5px 5px #888888;">'
                        '<h1 style="color: white; text-align: center;">Data Preprocessing</h1>'
                        '<p style="color: white; text-align: center; font-size: 18px;">This section allows you to clean and preprocess your data.</p>'
                        '</div>', unsafe_allow_html=True)
            st.write("")
            
            st.write("Data preprocessing is an essential step in data analysis. It involves cleaning, transforming, and organizing your data to make it suitable for analysis and modeling.")
            st.write("In this section, you can perform data preprocessing on your input data and generate processed data.")
            st.write("")
            
            if self.raw_data is not None:
                st.markdown('<div style="text-align: center"><h4> 📊 "Unleash the Magic of Your Data: Your Input, Your Power 💫" 📈 </div>', unsafe_allow_html=True)
                st.dataframe(self.raw_data)

                st.markdown('<div style="text-align: center"><h4> 🔮 "Revealing Insights: Your Processed Data, Your Discoveries ✨"</div>', unsafe_allow_html=True)
                st.dataframe(self.processed_data)

                eda = pd.read_csv("Pre_processed_data/processed_data.csv")
                # Create a profile report object
                pr = ProfileReport(eda)

                if st.button("Click For Cleaned Data EDA"):
                    st_profile_report(pr)

            else:
                st.write("Upload your dataset first.")
        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")

