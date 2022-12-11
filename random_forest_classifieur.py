


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold 
from sklearn.metrics import accuracy_score as ac
from sklearn.metrics import  precision_score
from sklearn.metrics import  f1_score
from sklearn.ensemble import RandomForestClassifier




class RandForestClassifier(object):


    def __init__(self, depth = 1,  min_samples_split = 2, n_estimators = 100, criterion = "gini"):
        self.depth = depth
        self.random_forest = RandomForestClassifier(random_state=0,criterion=criterion,n_estimators=n_estimators, max_depth=depth, min_samples_split=min_samples_split,)


    def train(self,data_train,labels):
        self.random_forest = self.random_forest.fit(data_train, labels)
        
    
    def cross_validation(self,data_train,labels,depth_list,estimators_list,split_list):
        grilleRecherche = list()


        for grid_split in split_list:
            for grid_depth in depth_list:
                for grid_estimator in estimators_list:
                    grilleRecherche.append({"depth": grid_depth,"n_estimators":grid_estimator, "min_samples_split": grid_split})
        
        k = 5

        kf = KFold(n_splits=k, random_state=None, shuffle=True)
        accuracy_hyperparameter = {}
        precision_hyperparameter = {}
        f1_hyperparameter = {}
        for hyper in range(len(grilleRecherche)):
            
            depth = grilleRecherche[hyper]["depth"]
            n_estimators = grilleRecherche[hyper]["n_estimators"]
            min_samples_split = grilleRecherche[hyper]["min_samples_split"]

            random_forest_classifier = RandForestClassifier(depth = depth, n_estimators=n_estimators,min_samples_split=min_samples_split)
            accuracy = 0
            precision = 0
            f1 = 0
            for train_index, validation_index in kf.split(labels):

                X_train, X_validation = data_train[train_index], data_train[validation_index]
                y_train, y_validation = labels[train_index], labels[validation_index]

                random_forest_classifier.train(X_train,y_train) # train the model

                predictions = random_forest_classifier.predict(X_validation)
                accuracy += ac(y_validation,predictions) # compute the accuracy
                precision += precision_score(y_validation,predictions, average = "micro") # compute the precision
                f1 += f1_score(y_validation,predictions, average = "micro") # compute the precision
            
            accuracy_hyperparameter[hyper] = accuracy / k
            precision_hyperparameter[hyper] = precision / k
            f1_hyperparameter[hyper] = f1 / k
            
        hyper = max(accuracy_hyperparameter, key=accuracy_hyperparameter.get)
        print("Best hyperparameters:", grilleRecherche[hyper])

        self.random_forest = RandomForestClassifier(random_state=0, max_depth=grilleRecherche[hyper]["depth"])
        self.random_forest = self.random_forest.fit(data_train, labels)

        return accuracy_hyperparameter[hyper], f1_hyperparameter[hyper], precision_hyperparameter[hyper]


    def predict(self, X):

        if len(X.shape) == 1:  # Predict on one sample
            class_label = np.array([self.random_forest.predict(X)])
            return class_label

        elif len(X.shape) == 2:  # Predict on multiple samples
            class_label = []
            for index in range(X.shape[0]):
                class_label.append(self.random_forest.predict(X[index].reshape(1,-1))[0])
            return class_label
