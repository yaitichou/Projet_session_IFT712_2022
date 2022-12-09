

import numpy as np
import pandas as pd
from IPython.display import display

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report


class NNClassifier(object):
    """
    Class to implement a neural network model based on sklearn Multi Layers Perceptron (MLP).

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

        self.estimator = MLPClassifier(max_iter=800)
        self.scorers = scorers
        self.hyper_search = None

    def train_default(self, verbose=False):
        """
        Train the model with default sklearn parameters.

        Return the accuracy for train and validation data.
        And fit self.estimator with the best scorer
        """
        self.estimator.fit(self.x_train, self.y_train)

        pred = self.estimator.predict(self.x_train)
        accu_train = accuracy_score(self.y_train, pred)

        pred_val = self.estimator.predict(self.x_val)
        accu_val = accuracy_score(self.y_val, pred_val)

        if verbose:
            print('Accuracy train: {:.3%}'.format(accu_train))
            print('Accuracy validation: {:.3%}'.format(accu_val))

        return accu_train, accu_val

    def train_hyperparameter(self, estimator_params, random_search=True, verbose=False):
        """
        Train the model with a grid search and a cross validation

        Inputs:
        - grid_search_params (dict) -- Dictionnary with values to test in the grid search. If not given, it will use the estimator's default values.
        - random_search (bool), default=True -- If True use the Randimized Search, 
                if False search all combination of parameters (take longer time).

        Returns:
        - training accuracy
        - validation accuracy
        And
        - fit self.estimator with the best scorer
        """

        # Grid search init with kfold
        searching_params = {
            "scoring": self.scorers,
            "refit": "Accuracy",
            "cv": KFold(n_splits=5, shuffle=True),
            "return_train_score": True,
            "verbose": int(verbose),
            "n_jobs": 4}

        if random_search:
            if verbose:
                print("Using randomized search:")
            self.hyper_search = RandomizedSearchCV(self.estimator, estimator_params).set_params(**searching_params)
        else:
            if verbose:
                print("Using complet search:")
            self.hyper_search = GridSearchCV(self.estimator, estimator_params).set_params(**searching_params)

        # Search hyper parameters
        self.hyper_search.fit(self.x_train, self.y_train)

        # Save best estimator
        self.estimator = self.hyper_search.best_estimator_

        # Predictions on train and validation data
        pred_train = self.hyper_search.predict(self.x_train)
        pred_val = self.hyper_search.predict(self.x_val)

        # Train and validation accuracy
        accu_train = accuracy_score(self.y_train, pred_train)
        accu_val = accuracy_score(self.y_val, pred_val)

        if verbose:
            print()
            print('Best cross val accuracy : {}'.format(self.hyper_search.best_score_))
            print('Best estimator:\n{}'.format(self.hyper_search.best_estimator_))
            print()
            print('Accuracy train: {:.3%}'.format(accu_train))
            print('Accuracy validation: {:.3%}'.format(accu_val))

        return accu_train, accu_val

    def predict(self, X):
        """
        Use the trained model to predict the sample's class.
        X: A list containing one or many samples.

        Returns a encoded class label for each sample.
        """
        return self.estimator.predict(X)


