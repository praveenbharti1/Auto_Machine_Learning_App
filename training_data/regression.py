import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import streamlit as st
import xgboost as xgb
import pandas as pd
import joblib
import os
from Logger import CustomLogger


class MyRegressionModel:
    def __init__(self):
        self.linear = LinearRegression()
        self.lasso = Lasso()
        self.ridge = Ridge()
        self.elastic_net = ElasticNet()
        self.decision_tree = DecisionTreeRegressor()
        self.random_forest = RandomForestRegressor()
        self.svm = SVR()
        self.k_neighbors = KNeighborsRegressor()
        self.adaboost = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4))
        self.gradient_descent = None  # Placeholder for gradient descent
        self.xgboost = xgb.XGBRegressor()
        self.log = CustomLogger("log.log")

    def train_linear_regression(self, X, y):
        try:
            self.model = self.linear
            self._train_and_evaluate(X, y)
            self.log.log_info(f"Training of Linear Regression model is complete.")
            if not os.path.exists("Model/Regression_model"):
                os.makedirs("Model/Regression_model")
            joblib.dump(self.model,
                        f"Model/Regression_model/{self.model.__class__.__name__}.save")
        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")

    def train_lasso_regression(self, X, y):
        try:
            self.model = self.lasso
            self._train_and_evaluate(X, y)
            self.log.log_info(f"Training of Lasso Regression model is complete.")
            if not os.path.exists("Model/Regression_model"):
                os.makedirs("Model/Regression_model")
            joblib.dump(self.model,
                        f"Model/Regression_model/{self.model.__class__.__name__}.save")
        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")

    def train_ridge_regression(self, X, y):
        try:
            self.model = self.ridge
            self._train_and_evaluate(X, y)
            self.log.log_info(f"Training of Ridge Regression model is complete.")
            if not os.path.exists("Model/Regression_model"):
                os.makedirs("Model/Regression_model")
            joblib.dump(self.model,
                        f"Model/Regression_model/{self.model.__class__.__name__}.save")
        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")

    def train_elastic_net_regression(self, X, y):
        try:
            self.model = self.elastic_net
            self._train_and_evaluate(X, y)
            self.log.log_info(f"Training of Elastic net Regression model is complete.")
            if not os.path.exists("Model/Regression_model"):
                os.makedirs("Model/Regression_model")
            joblib.dump(self.model,
                        f"Model/Regression_model/{self.model.__class__.__name__}.save")
        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")

    def train_decision_tree_regression(self, X, y):
        try:
            self.model = self.decision_tree
            self._train_and_evaluate(X, y)
            self.log.log_info(f"Training of Decision tree Regression model is complete.")
            if not os.path.exists("Model/Regression_model"):
                os.makedirs("Model/Regression_model")
            joblib.dump(self.model,
                        f"Model/Regression_model/{self.model.__class__.__name__}.save")
        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")

    def train_random_forest_regression(self, X, y):
        try:
            self.model = self.random_forest
            self._train_and_evaluate(X, y)
            self.log.log_info(f"Training of Random forest Regression model is complete.")
            if not os.path.exists("Model/Regression_model"):
                os.makedirs("Model/Regression_model")
            joblib.dump(self.model,
                        f"Model/Regression_model/{self.model.__class__.__name__}.save")
        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")

    def train_svm_regression(self, X, y):
        try:
            self.model = self.svm
            self._train_and_evaluate(X, y)
            self.log.log_info(f"Training of SVM Regression model is complete.")
            if not os.path.exists("Model/Regression_model"):
                os.makedirs("Model/Regression_model")
            joblib.dump(self.model,
                        f"Model/Regression_model/{self.model.__class__.__name__}.save")
        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")

    def train_k_neighbors_regression(self, X, y):
        try:
            self.model = self.k_neighbors
            self._train_and_evaluate(X, y)
            self.log.log_info(f"Training of k neighbour Regression model is complete.")
            if not os.path.exists("Model/Regression_model"):
                os.makedirs("Model/Regression_model")
            joblib.dump(self.model,
                        f"Model/Regression_model/{self.model.__class__.__name__}.save")
        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")

    def train_adaboost_regression(self, X, y):
        try:
            self.model = self.adaboost
            self._train_and_evaluate(X, y)
            self.log.log_info(f"Training of adaboost Regression model is complete.")
            if not os.path.exists("Model/Regression_model"):
                os.makedirs("Model/Regression_model")
            joblib.dump(self.model,
                        f"Model/Regression_model/{self.model.__class__.__name__}.save")
        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")

    def train_gradient_descent(self, X, y):
        self.model = self.gradient_descent
        # Implement gradient descent training here
        pass

    def train_xgboost_regression(self, X, y):
        try:
            self.model = self.xgboost
            self._train_and_evaluate(X, y)
            self.log.log_info(f"Training of XGBoost Regression model is complete.")
            if not os.path.exists("Model/Regression_model"):
                os.makedirs("Model/Regression_model")
            joblib.dump(self.model,
                        f"Model/Regression_model/{self.model.__class__.__name__}.save")
        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")

    def _train_and_evaluate(self, X, y):
        try:
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

            # Train the model on the training data
            self.model.fit(X_train, y_train)

            # Make predictions on the test data
            y_pred = self.model.predict(X_test)

            # Calculate the model's performance
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Print the model's performance metrics
            self.log.log_info(f"Algorithm: {self.model.__class__.__name__}")
            self.log.log_info(f"Mean Squared Error:, {str(mse)}")
            self.log.log_info(f"R-squared:, {str(r2)}")
        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")



    def display_accuracies(self, X, y):
        try:
            st.write("# Regression Model Accuracies")

            # Create empty lists to store model names and accuracy metrics
            model_names = []
            mse_scores = []
            r2_scores = []

            # Define a list of trained regression models
            regression_models = [
                ("Linear Regression", self.train_linear_regression),
                ("Lasso Regression", self.train_lasso_regression),
                ("Ridge Regression", self.train_ridge_regression),
                ("Elastic Net Regression", self.train_elastic_net_regression),
                ("Decision Tree Regression", self.train_decision_tree_regression),
                ("Random Forest Regression", self.train_random_forest_regression),
                ("SVM Regression", self.train_svm_regression),
                ("K-Neighbors Regression", self.train_k_neighbors_regression),
                ("Adaboost Regression", self.train_adaboost_regression),
                ("XGBoost Regression", self.train_xgboost_regression),
            ]

            for model_name, train_method in regression_models:
                # Train the model
                train_method(X, y)

                # Make predictions
                y_pred = self.model.predict(X)

                # Calculate the model's performance on the entire dataset
                mse = mean_squared_error(y, y_pred)
                r2 = r2_score(y, y_pred)

                model_names.append(model_name)
                mse_scores.append(mse)
                r2_scores.append(r2)

            # Create a DataFrame to display the accuracies in a table
            accuracy_data = pd.DataFrame({
                "Model": model_names,
                "Mean Squared Error (MSE)": mse_scores,
                "R-squared (R2) Score": r2_scores,
            })

            # Display the table in the Streamlit app
            st.table(accuracy_data)
        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")



        # Plot the original data points and the regression line (for linear models)
        # if isinstance(self.model, (LinearRegression, Lasso, Ridge, ElasticNet)):
        #     plt.scatter(X_test, y_test, color='blue')
        #     plt.plot(X_test, y_pred, color='red', linewidth=3)
        #     plt.xlabel('X')
        #     plt.ylabel('y')
        #     plt.title(f'{self.model.__class__.__name__} Regression')
        #     plt.show()
