import streamlit as st
import pandas as pd
import joblib
from data_preprocessing.data_imputation import DataProcessing
import pandas as pd
from Logger import CustomLogger
import os
import warnings
warnings.filterwarnings('ignore')




class AutoMlVisual:
    def __init__(self):
        self.dp = DataProcessing()
        self.scaler = joblib.load(r"Model/scaler.save")
        self.label_encoder = joblib.load(r"Model/label_encode.joblib")
        self.log = CustomLogger("log.log")
        self.imputed_data = None
        self.label_encod = None

    def user_input_features(self):
        try:
            selected_file = pd.read_csv("SelectedTask/selectedtask.csv")
            selected_task = str(selected_file.iloc[0,0])
        
            if selected_task == "Regression":

                processed_data = pd.read_csv("Pre_processed_data/processed_data.csv")
                scaling_df = pd.read_csv("Pre_Trained_Data/scaled_data.csv")

                final_cols = scaling_df.columns.intersection(processed_data.columns)
                data = processed_data[final_cols]

                files = [os.listdir(r"Model/Regression_model")] 
                model_name = [] 
                for i in files[0]:
                    model_name.append(i.split(".")[0])
                
                self.selected_model = st.selectbox(options= model_name, label = "Choose Model For Getting Predicted Value")
                
                file_path = os.path.join(r"Model/Regression_model/", f"{self.selected_model}.save")
                self.model = joblib.load(str(file_path))

                cat_columns = [
                    column for column in data.columns if data[column].dtype == "object"]

                self.imputed_data = {}
                for col in data.columns:
                    if col in cat_columns:
                        self.imputed_data[col] = st.sidebar.selectbox(
                            f"Select {col}", data[col].unique())
                    else:
                        min_val = data[col].min()
                        max_val = data[col].max()
                        self.imputed_data[col] = st.sidebar.slider(
                            f"Select {col}", min_val, max_val)

                
                
                self.imputed_data = pd.DataFrame(self.imputed_data, index=[0])

                if col in cat_columns:
                    self.imputed_data[col] = self.label_encoder.transform(self.imputed_data[col])
                else:
                    self.log.log_info("all are non categorical columns")

                self.pred = self.model.predict(self.imputed_data)
            
        
            if selected_task == "Classification":

                raw_data = pd.read_csv(r"Modified_Data/modified_data.csv")
                scaling_df = pd.read_csv(r"Pre_Trained_Data/scaled_data.csv")

                col_intrstn = raw_data.columns.intersection(scaling_df.columns) 

                
                self.data_raw = raw_data[col_intrstn]
                st.dataframe(raw_data)
                
                

                cat_columns = [
                    column for column in self.data_raw.columns if self.data_raw[column].dtype == "object"]
                

                self.imputed_data = {}
                for col in self.data_raw.columns:
                    if col in cat_columns:
                        self.imputed_data[col] = st.sidebar.selectbox(
                            f"Select {col}", self.data_raw[col].unique())
                    else:
                        min_val = self.data_raw[col].min()
                        max_val = self.data_raw[col].max()
                        self.imputed_data[col] = st.sidebar.slider(
                            f"Select {col}", min_val, max_val)
                                
                self.imp_data = pd.DataFrame(self.imputed_data, index=[0])
                st.write('### User Input Data')
                st.dataframe(self.imp_data)

                st.write("""
                        ### Select The Model For Predicton
                        """)
                
                files = [os.listdir(r"Model/Classification_Model")] 
                model_name = [] 
                for i in files[0]:
                    model_name.append(i.split(".")[0])
                
                self.selected_model = st.selectbox(options= model_name, label = "Choose Model For Getting Predicted Value")
                
                file_path = os.path.join(r"Model/Classification_Model/", f"{self.selected_model}.save")
                self.model = joblib.load(str(file_path))

                for column in self.imp_data.select_dtypes(include=['object']).columns:
                    if column in self.label_encoder:
                        label_encoder_for_column = self.label_encoder[column]
                        self.imp_data[column] = label_encoder_for_column.transform(self.imp_data[column])
                        

                
                self.scale_data = self.scaler.transform(self.imp_data)

                st.subheader('Encoded Data')
                st.dataframe(self.imp_data)
                
                self.pred = self.model.predict(self.scale_data)
                st.subheader("prediction")
                st.dataframe(self.pred)
                
                self.prediction_proba = self.model.predict_proba(self.scale_data)
                st.subheader('Prediction Probality')
                st.dataframe(self.prediction_proba)
        except Exception as e:
            self.log.log_info(f"this is error and your error is :: {str(e)} ")
          