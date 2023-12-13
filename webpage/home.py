import os
import pandas as pd
import streamlit as st
from pymongo.mongo_client import MongoClient
from Logger import CustomLogger
import warnings

warnings.filterwarnings('ignore')

# Initialize connection to MongoDB using st.secrets
# mongo_username = st.secrets["secrets"]["username"]
# mongo_password = st.secrets["secrets"]["password"]


class Home_Page:
    def __init__(self):
        self.file = None
        self.task = None
        self.selected_task = None
        self.data = None
        self.raw_data = None
        self.log = CustomLogger("log.log")

    st.cache_data(ttl=600)
    def upload_data(self):
        try:
            st.markdown(
                """
                <style>
                    body {
                        background-size: cover;
                        font-family: 'Arial', sans-serif;
                        color: #333;
                    }
                    .logo {
                        text-align: center;
                    }
                    .slogan {
                        font-size: 18px;
                        font-weight: bold;
                        color: #fff;
                        background-color: #333;
                        padding: 5px 10px;
                        border-radius: 5px;
                    }
                </style>
                """,
                unsafe_allow_html=True
            )

            st.markdown('<div style="text-align: center"><h1> Auto Machine Learning App </div>', unsafe_allow_html=True)
            st.write("\n")
            st.markdown('<div style="text-align: center"><h4> 🌟 Welcome to AutoMachineLearningApp! 🚀 </div>', unsafe_allow_html=True)
            st.markdown('<div class="slogan" style="text-align: center">✨**INNOVATE**...🔗**INTEGRATE**...🌈**ILLUMINATE**!!! 🎉</div>', unsafe_allow_html=True)

            st.markdown('<div class="slogan" style="text-align: center">Automate Your Machine Learning Tasks with Ease.</div>', unsafe_allow_html=True)
            st.write("")
            st.markdown('''
                
                🌟 Welcome to Auto Machine Learning! 🤖, your gateway to effortless data-driven insights 📊. 
                Harness the power of automated machine learning to make sense of your data, 
                whether you're tackling classification problems or regression challenges.
                ## Key Features

                - 🚀 Automated machine learning (AutoML)
                - 📊 Classification and regression tasks
                - 📁 Easy data upload and preprocessing
                - 🧙 Intuitive interface for model selection
                - 📈 Visualize and interpret results
                ''', unsafe_allow_html=True)
            st.markdown('Ready to get started? Get started by uploading your dataset.')

            

            # Option to choose between file upload and table selection
            option = st.radio("Choose an option:", ["Upload a file", "Select a table from Sample Data"], horizontal=True)

            if option == "Upload a file":
                # File uploader
                self.file = st.file_uploader("Let's automate your ML tasks")

                if self.file is not None:
                    # Check the file format here, e.g., if it's a CSV file
                    if self.file.type == "application/vnd.ms-excel" or self.file.type == "text/csv":
                        if not os.path.exists("Raw_data"):
                            # Create the directory if it doesn't exist
                            os.makedirs("Raw_data")
                        df = pd.read_csv(self.file, index_col=None)
                        self.raw_data = df
                        df.to_csv("Raw_data/raw_file.csv", index=False)
                        self.log.log_info("Data uploaded and saved to 'Raw_data/raw_file.csv'")
                    else:
                        st.error("Unsupported file format. Please upload a CSV file.")

                    st.write('Your Uploaded Data:')
                    self.data = pd.read_csv("Raw_data/raw_file.csv")

            elif option == "Select a table from Sample Data":
                client = MongoClient(f"mongodb+srv://database:JwkMgMqhEniSNHuX@cluster0.xgvrey3.mongodb.net/?retryWrites=true&w=majority")
                db = client['your_mongo_database']
                table_list = db.list_collection_names()
                st.write(table_list)
                table_name  = [list for list in table_list]

                selected_table = st.selectbox("Select a table", table_name)
                
                # Display the selected table name
                st.write(f"You selected: {selected_table}")

                dataframe = list(db[selected_table].find())
                df = pd.DataFrame(dataframe)
                df = df.iloc[:,1:]
                # Find and drop unnamed columns
                unnamed_columns = [col for col in df.columns if 'Unnamed: 0' in col]

                if unnamed_columns:
                    df = df.drop(columns=unnamed_columns)

                # Display the selected DataFrame
                st.header('Selected Table Data:')
                st.write(df)
                
                self.raw_data = df
                df.to_csv("Raw_data/raw_file.csv", index=False)
                self.log.log_info("Data uploaded and saved to 'Raw_data/raw_file.csv'")
        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")

    def select_task(self):
        try:
            home_page = Home_Page()
            selected_task = home_page.get_selected_task(display_radio=True)
            return selected_task
        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")

    def get_selected_task(self, display_radio=True):
        try:
            if display_radio:
                task = st.radio(
                    "Select Task:", ["Classification", "Regression"])
                if task is not None:  # Check if task is not None
                    return task  # Return the selected task
            return self.selected_task  # Return the existing selected task
        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")

    def run(self):
        try:
            st.write("""
            # Auto Machine Learning
            """)
        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")


