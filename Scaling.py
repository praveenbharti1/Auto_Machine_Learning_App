import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler
from Logger import CustomLogger


class DataScaler:
    def __init__(self, data):
        self.data = data
        self.scaler = StandardScaler()
        self.scaled_data = None
        self.log = CustomLogger("log.log")

    def scale_data(self):
        try:
            if self.data is not None:
                self.scaled_data = self.scaler.fit_transform(self.data)
                scaled_data = pd.DataFrame(
                    self.scaled_data, columns=self.data.columns)
                self.log.log_info("Data Scaling Has Done.....")
                
                if not os.path.exists("Model"):
                    os.makedirs("Model")
                    os.makedirs("Pre_Trained_Data")
                joblib.dump(self.scaler, "Model/scaler.save")
                scaled_data.to_csv("Pre_Trained_Data/scaled_data.csv", index=False)
                self.log.log_info("Scaling Model Has Been Saved....")
                return scaled_data
            else:
                self.log.log_info(
                    "No data provided for scaling. Please set the 'data' attribute.")
        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")
