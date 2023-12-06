import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import streamlit as st
import pandas as pd
import joblib
import os
from Logger import CustomLogger



class AutoClassifier:
    def __init__(self, test_size=0.25, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.log = CustomLogger("log.log")

    def train_test_split(self, X, y, test_size=0.25, random_state=42):
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")


    def train_random_forest(self, X_train, y_train):
        try:
            self.model = RandomForestClassifier()
            self.model.fit(X_train, y_train)
            self.log.log_info(f"Fit RandomForestClassifier on X_train and Y_train")

            if not os.path.exists("Model/Classification_Model"):
                os.makedirs("Model/Classification_Model")
            joblib.dump(self.model, f"Model/Classification_Model/{self.model.__class__.__name__}.save")
        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")


    def train_knn(self, X_train, y_train, n_neighbors=5):
        try:
            self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
            self.model.fit(X_train, y_train)
            self.log.log_info(f"Fit KNeighborsClassifier on X_train and Y_train")

            if not os.path.exists("Model/Classification_Model"):
                os.makedirs("Model/Classification_Model")
            joblib.dump(self.model, f"Model/Classification_Model/{self.model.__class__.__name__}.save")
        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")
 

    def train_naive_bayes(self, X_train, y_train):
        try:
            self.model = GaussianNB()
            self.model.fit(X_train, y_train)
            self.log.log_info(f"Fit GaussianNB on X_train and Y_train")

            if not os.path.exists("Model/Classification_Model"):
                os.makedirs("Model/Classification_Model")
            joblib.dump(self.model, f"Model/Classification_Model/{self.model.__class__.__name__}.save")
        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")


    def train_decision_tree(self, X_train, y_train):
        try:
            self.model = DecisionTreeClassifier()
            self.model.fit(X_train, y_train)
            self.log.log_info(f"Fit DecisionTreeClassifier on X_train and Y_train")

            if not os.path.exists("Model/Classification_Model"):
                os.makedirs("Model/Classification_Model")
            joblib.dump(self.model, f"Model/Classification_Model/{self.model.__class__.__name__}.save")
        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")


    def train_logistic_regression(self, X_train, y_train):
        try:
            self.model = LogisticRegression()
            self.model.fit(X_train, y_train)
            self.log.log_info(f"Fit LogisticRegression on X_train and Y_train")

            if not os.path.exists("Model/Classification_Model"):
                os.makedirs("Model/Classification_Model")
            joblib.dump(self.model, f"Model/Classification_Model/{self.model.__class__.__name__}.save") 
        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")
         

    def train_svm(self, X_train, y_train):
        try:
            self.model = SVC(probability=True)
            self.model.fit(X_train, y_train)
            self.log.log_info(f"Fit SupportVectorClassifier on X_train and Y_train")

            if not os.path.exists("Model/Classification_Model"):
                os.makedirs("Model/Classification_Model")
            joblib.dump(self.model, f"Model/Classification_Model/{self.model.__class__.__name__}.save")
        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")

    def train_adaboost(self, X_train, y_train):
        try:
            base_classifier = DecisionTreeClassifier(max_depth=1)
            self.model = AdaBoostClassifier(base_classifier, n_estimators=50)
            self.model.fit(X_train, y_train)
            self.log.log_info(f"Fit AdaBoostClassifier on X_train and Y_train")

            if not os.path.exists("Model/Classification_Model"):
                os.makedirs("Model/Classification_Model")
            joblib.dump(self.model, f"Model/Classification_Model/{self.model.__class__.__name__}.save") 
        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")

    def train_and_evaluate(self, clf, X_train, X_test, y_train, y_test):
            try:
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                return accuracy
            except Exception as e:
                self.log.log_info(f"this is error and your error is{str(e)} ")




    def classification_model_accuracy(self, X, y):
        try:
            st.title("Classifier Accuracy Comparison")

            test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.25)
            random_state = st.sidebar.slider("Random State", 0, 100, 42)

            classifiers = [
                ("RandomForest", RandomForestClassifier()),
                ("KNN", KNeighborsClassifier()),
                ("NaiveBayes", GaussianNB()),
                ("DecisionTree", DecisionTreeClassifier()),
                ("LogisticRegression", LogisticRegression()),
                ("SVM", SVC(probability=True)),
                ("AdaBoost", AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50))
            ]

            accuracies = []

            for clf_name, clf in classifiers:
                X_train, X_test, y_train, y_test = self.train_test_split(X, y, test_size, random_state)
                accuracy = self.train_and_evaluate(clf, X_train, X_test, y_train, y_test)
                accuracies.append((clf_name, accuracy))

            df = pd.DataFrame(accuracies, columns=['Classifier', 'Accuracy'])
            st.dataframe(df)
        except Exception as e:
            self.log.log_info(f"this is error and your error is{str(e)} ")




    # labels, scores = zip(*accuracies)

    # Plot the accuracies
    # plt.figure(figsize=(10, 6))
    # plt.barh(labels, scores, color='skyblue')
    # plt.xlabel('Accuracy')
    # plt.title('Classifier Accuracy Comparison')
    # plt.show()
