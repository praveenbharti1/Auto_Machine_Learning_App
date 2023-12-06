import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from webpage.home import Home_Page
from webpage.EDA import Ex_Data_Ana
from webpage.Data_process import DataPreprocessing
from webpage.feature_select import FeatureSelection
from Scaling import DataScaler
from training_linear_dataset import RegressionTrainer
from training_class_models import ClassificationTrainer
from data_preprocessing.data_clean import DataClean
from Final_Output import AutoMlVisual
from training_data.regression import MyRegressionModel
from webpage.about import About
from training_data.classification import AutoClassifier
from Logger import CustomLogger
import joblib
import os


st.set_page_config(page_title="Auto Machine Learning", initial_sidebar_state="expanded", layout="centered")

class AutoML:    
        
        def __init__(self) -> None:
            self.log = CustomLogger("log.log")

        def run(self):
            try:
                with st.sidebar:
                    st.image("https://www.onepointltd.com/wp-content/uploads/2019/12/ONE-POINT-01-1.png")
                    app = option_menu(
                        menu_title='Auto Machine Learning',
                        options=['Home', 'EDA', 'Data Pre-Processing', 'Feature Engineering', 'Data Visualisation','About'],
                        icons=['house-fill', 'bar-chart-fill', 'ui-radios', 'motherboard', 'motherboard', 'info-circle-fill'],
                        menu_icon='chat-text-fill',
                        default_index=0,
                        styles={
                            "container": {"padding": "5!important", "background-color": 'black'},
                            "icon": {"color": "white", "font-size": "23px"},
                            "nav-link": {"color": "white", "font-size": "20px", "text-align": "left", "margin": "0px",
                                        "--hover-color": "blue"},
                            "nav-link-selected": {"background-color": "#02ab21"},
                        }
                    )
                
                if app == "Home":
                    home_app = Home_Page()
                    home_app.upload_data()
                    selected_task =home_app.select_task()
                    select_data = pd.DataFrame([[selected_task]])
                    # Create a directory if it doesn't exist
                    if not os.path.exists('SelectedTask'):
                        os.makedirs('SelectedTask')
                    select_data.to_csv("SelectedTask/selectedtask.csv", index=False)
                    if home_app.data is not None:
                        st.dataframe(home_app.data)
                    

                    
                if app == "EDA":
                    # Define the file path to dataset
                    filepath = "Raw_data/raw_file.csv"  

                    # Create an instance of the Ex_Data_Ana class and run the exploratory data analysis
                    data_analyzer = Ex_Data_Ana(filepath)
                    data_analyzer.exploratory_data_analysis()   

                if app == 'Data Pre-Processing':
                    # Define the file path to dataset
                    filepath = "Raw_data/raw_file.csv"  

                    # Initialize the DataClean class
                    data_cleaner = DataClean(filepath)
                    data_cleaner.save_cleaned_data()

                    
                    preprocess = DataPreprocessing()
                    preprocess.process_data()

                
                if app == 'Feature Engineering':
                    # Create an instance of the DataFrameColumnSelector class
                    column_selector = FeatureSelection()
                    
                    # Display the app
                    column_selector.display()
                
                    # Get X and y
                    X, y = column_selector.get_X_y()
                    if X is not None:
                    # Display correlation heatmap and optionally drop highly correlated columns
                        updated_X = column_selector.display_correlation_heatmap()

                        if updated_X is not None:
                            st.write("Updated X after dropping columns:")
                            st.write(updated_X)

                    # Create an instance of the DataScaler class
                        scaler = DataScaler(X)

                        # Scale the data using the StandardScaler
                        scaled_data = scaler.scale_data()
                        st.write("""### your data is scaled now""")
                        st.dataframe(scaled_data)

                        
                        X = pd.read_csv("Pre_Trained_Data/scaled_data.csv")
                        y = pd.read_csv("Pre_Trained_Data/dependent.csv")
                        selected_file = pd.read_csv("SelectedTask/selectedtask.csv")
                        selected = str(selected_file.iloc[0,0])
                    

                        
                    
                        if selected is not None:
                            if selected == "Classification":
                                trainer = ClassificationTrainer()
                                trainer.train_and_evaluate_models(X, y)
                                

                            elif selected == "Regression":
                                trainer = RegressionTrainer()
                                trainer.train_and_evaluate_models(X, y)
                        
                        
                        if selected == 'Regression':
                            # Instantiate your MyRegressionModel
                            my_model = MyRegressionModel()
                            # Display accuracies in Streamlit
                            my_model.display_accuracies(X, y)
                        else:
                            my_model = AutoClassifier()
                            my_model.classification_model_accuracy(X, y)
            
                        
                if app == 'Data Visualisation':
                    selected_file = pd.read_csv("SelectedTask/selectedtask.csv")
                    selected_task = str(selected_file.iloc[0,0])
            
                    if selected_task == "Regression":
                        st.write("""
                        # AUTOMATIC MACHINE LEARNING          
                        """)

                        st.sidebar.header('User Input Features')

                        user_input = AutoMlVisual()

                        user_input.user_input_features()

                        st.write("""
                                ## User Input Features
                                """)
                        
                        st.dataframe(user_input.imputed_data)

                        st.subheader('Predictions')
                        st.dataframe(user_input.pred)
                    else:
                        
                        st.write("""
                        # AUTOMATIC MACHINE LEARNING          
                        """)
                        st.sidebar.header('User Input Features')

                        user_input = AutoMlVisual()
                        user_input.user_input_features()

                     
                if app == "About": 
                    about_page = About()
                    about_page.aboutsection() 
                        
            except Exception as e:
                self.log.log_info(f"this is error and your error is :: {str(e)} ")

                                        

# create AutoML instance and run
if __name__ == '__main__':
    auto_ml = AutoML()
    auto_ml.run()

    