from training_data.regression import MyRegressionModel
from Logger import CustomLogger


class RegressionTrainer:
    def __init__(self):
        self.regression_model = MyRegressionModel()
        self.log = CustomLogger("log.log")

    def train_and_evaluate_models(self, X, y):
        try:
            self.log.log_info("Regression Model Training And Evaluation has Been Started Successfully")

            self.regression_model.train_linear_regression(X, y)
            self.log.log_info("Linear Regression Model Training And Evaluation has Been Done Successfully")

            self.regression_model.train_lasso_regression(X, y)
            self.log.log_info("Lasso Regression Model Training And Evaluation has Been Done Successfully")

            self.regression_model.train_ridge_regression(X, y)
            self.log.log_info("Ridge Regression Model Training And Evaluation has Been Done Successfully")

            self.regression_model.train_elastic_net_regression(X, y)
            self.log.log_info("Elastic Net Regression Model Training And Evaluation has Been Done Successfully")

            self.regression_model.train_decision_tree_regression(X, y)
            self.log.log_info("Decision Tree Model Training And Evaluation has Been Done Successfully")

            self.regression_model.train_random_forest_regression(X, y)
            self.log.log_info("Random Forest Regression Model Training And Evaluation has Been Done Successfully")

            self.regression_model.train_svm_regression(X, y)
            self.log.log_info("SVM Regression Model Training And Evaluation has Been Done Successfully")

            self.regression_model.train_k_neighbors_regression(X, y)
            self.log.log_info("KNN Regression Model Training And Evaluation has Been Done Successfully")

            self.regression_model.train_adaboost_regression(X, y)
            self.log.log_info("AdaBoost Regression Model Training And Evaluation has Been Done Successfully")

            self.regression_model.train_gradient_descent(X, y)  
            self.log.log_info("Gradient Descent Regression Model Training And Evaluation has Been Done Successfully")

            self.regression_model.train_xgboost_regression(X, y)
            self.log.log_info("XGBoost Regression Model Training And Evaluation has Been Done Successfully")

            self.log.log_info("all model is trained")
        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")

