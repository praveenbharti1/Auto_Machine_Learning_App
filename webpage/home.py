import os
import pandas as pd
import streamlit as st
from Logger import CustomLogger
import warnings
from sqlalchemy import create_engine, MetaData

warnings.filterwarnings('ignore')

class Home_Page:
    def __init__(self):
        self.file = None
        self.task = None
        self.selected_task = None
        self.data = None
        self.raw_data = None
        self.log = CustomLogger("log.log")

    def get_table_names(self):
        database_name = 'sample_data\sample_data'
        # SQLite Database and Engine
        engine = create_engine(f'sqlite:///{database_name}.db')

        # Reflect the database schema
        metadata = MetaData()
        metadata.reflect(bind=engine)

        # Get a list of table names in the database
        table_names = metadata.tables.keys()
        st.write(table_names)

        # Close the connection
        engine.dispose()

        return list(table_names)

    def upload_data(self):
        try:
            st.markdown(
                """
                <style>
                    body {
                        background-image: url('webpage\asset\ketkibackgroundimg.jpg');
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

            # Streamlit app
            st.title("SQLite Table List Viewer")

            # Option to choose between file upload and table selection
            option = st.radio("Choose an option:", ["Upload a file", "Select a table from Sample Data"])

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
                
                db_name = r'sample_data\sample_data'
                tables = self.get_table_names()
                
                
                # Display a dropdown with the list of table names
                selected_table = st.selectbox("Select a table", tables)
                

                # Display the selected table name
                st.write(f"You selected: {selected_table}")

                # Convert selected table to DataFrame
                engine = create_engine(f'sqlite:///{db_name}.db')
                selected_df = pd.read_sql_table(selected_table, engine)

                # Display the selected DataFrame
                st.write('Selected Table Data:')
                st.write(selected_df)

                self.raw_data = selected_df
                selected_df.to_csv("Raw_data/raw_file.csv", index=False)
                self.log.log_info("Data uploaded and saved to 'Raw_data/raw_file.csv'")

                

                
                # Close the connection
                engine.dispose()
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


