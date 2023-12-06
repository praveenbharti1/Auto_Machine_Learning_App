from training_data.classification import AutoClassifier
from Logger import CustomLogger


class ClassificationTrainer:
    def __init__(self):
        self.classification_model = AutoClassifier()
        self.log = CustomLogger("log.log")

    def train_and_evaluate_models(self, X, y):
        try:
            X_train, X_test, y_train, y_test = self.classification_model.train_test_split(
                X, y)
            self.log.log_info(
                "Model has started train test split for classification")

            self.classification_model.train_random_forest(X_train, y_train)
            self.log.log_info("Random Forest Model Training And Evaluation has Been Done Successfully")

            self.classification_model.train_knn(X_train, y_train)
            self.log.log_info("KNN Model Training And Evaluation has Been Done Successfully")
            
            self.classification_model.train_naive_bayes(X_train, y_train)
            self.log.log_info("Naive Bayes Model Training And Evaluation has Been Done Successfully")

            self.classification_model.train_decision_tree(X_train, y_train)
            self.log.log_info("Decision Tree Model Training And Evaluation has Been Done Successfully")

            self.classification_model.train_logistic_regression(X_train, y_train)
            self.log.log_info("Logistic Rgression Model Training And Evaluation has Been Done Successfully")

            self.classification_model.train_svm(X_train, y_train)
            self.log.log_info("SVC Model Training And Evaluation has Been Done Successfully")

            self.classification_model.train_adaboost(X_train, y_train)
            self.log.log_info("AdaBoost Model Training And Evaluation has Been Done Successfully")
            # self.classification_model.plot_accuracies(X, y)
            self.log.log_info("all model is trained")
        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")

