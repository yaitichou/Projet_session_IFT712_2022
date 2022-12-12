


import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier


class AdaboostClassifier(object):
    """
    Class to implement a adaboost model based on sklearn module ensemble.

    Parameters:
    - x_train (array) -- Array of features values to train on.
    - y_train (array) -- Array of true labels to train the model.
    - x_val (array) -- Array of features values to validate the model.
    - y_val (array) -- Array of true labels to validate the model.
    - class_names (array) -- Array of names to link with labels
    """

    def __init__(self, x_train, y_train, x_val, y_val, class_names, scorers):
        self.x_train = x_train
        self.y_train = y_train

        self.x_val = x_val
        self.y_val = y_val
        self.class_names = class_names

        self.num_features = x_train.shape[1]
        self.num_classes = class_names.shape

        self.estimator = MultiOutputClassifier(AdaBoostClassifier(), n_jobs=4)
        self.scorers = scorers

    def train_sans_grid(self):
        """
        Train the model without a grid search

        Returns a tuple for:
        - training accuracy
        - validation accuracy
        """
        adaboost = self.estimator
        adaboost.fit(self.x_train, self.y_train)
        pred = adaboost.predict(self.x_train)
        accura_tr = accuracy_score(self.y_train, pred)
        pred_val = adaboost.predict(self.x_val)
        accura_val = accuracy_score(self.y_val, pred_val)

        return accura_tr, accura_val

    def train(self, grid_search_params={}, random_search=True):
        """
        Train the model with a grid search and a cross validation

        Inputs:
        - grid_search_params (dict) -- Dictionnary with values to test in the grid search. If not given, it will use the estimator's default values.
        - random_search (bool), default=True -- If True use the Randimized Search, 
                if False search all combination of parameters (take longer time).

        Returns a tuple for:
        - training accuracy
        - validation accuracy
        - best estiimator
        - best score
        """
        # Grid search init with kfold
        searching_params = {
            "scoring": self.scorers,
            "refit": "Accuracy",
            "cv": KFold(n_splits=5, shuffle=True),
            "return_train_score": True,
            "n_jobs": 4,
            "verbose": 1}

        if random_search:
            print("Using randomized search:")
            search_g = RandomizedSearchCV(self.estimator, grid_search_params).set_params(**searching_params)
        else:
            print("Using complet search:")
            search_g = GridSearchCV(self.estimator, grid_search_params).set_params(**searching_params)

        # Model training
        search_g.fit(self.x_train, self.y_train)

        # Save best estimator and print it with the best accuracy obtained through cross validation
        self.estimator = search_g.best_estimator_
        self.best_accuracy = search_g.best_score_
        self.hyper_search = search_g
        # Predictions on train and validation data
        pred_train = search_g.predict(self.x_train)
        pred_val = search_g.predict(self.x_val)

        # Train and validation accuracy
        acc_train = accuracy_score(self.y_train, pred_train)
        acc_val = accuracy_score(self.y_val, pred_val)

        return acc_train, acc_val, self.estimator, self.best_accuracy

    def predict(self, X):
        """
        Use the trained model to predict the sample's class.
        X: A list containing one or many samples.

        Returns a encoded class label for each sample.
        """
        class_label = self.estimator.predict(X)
        return class_label

        
