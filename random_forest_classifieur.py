


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold 
from sklearn.metrics import accuracy_score as ac

from sklearn.ensemble import RandomForestClassifier



class RandForestClassifier(object):


    def __init__(self, depth = 1, criterion = "gini"):
        self.depth = depth
        self.random_forest = RandomForestClassifier(random_state=0,criterion=criterion, max_depth=depth)


    def train(self,data_train,labels):
        self.random_forest = self.random_forest.fit(data_train, labels)
        
    
    def cross_validation(self,data_train,labels,depth_max):

        accuracy_hyperparameter = np.zeros(depth_max) 
        k = 10

        kf = KFold(n_splits=k, random_state=None, shuffle=True)

        for depth in range(1,depth_max):
            random_forest_classifier = RandForestClassifier(depth)
            accuracy = 0
            for train_index, validation_index in kf.split(labels):

                X_train, X_validation = data_train[train_index], data_train[validation_index]
                y_train, y_validation = labels[train_index], labels[validation_index]

                random_forest_classifier.train(X_train,y_train) # train the model

                predictions = random_forest_classifier.predict(X_validation)
                accuracy += ac(y_validation,predictions) # compute the accuracy

            accuracy_hyperparameter[depth-1] = accuracy / k
            
        best_depth = np.argmax(accuracy_hyperparameter) + 1

        print("best_depth >> "+str(best_depth )+" accuracy >> "+str(accuracy_hyperparameter[best_depth - 1]))

        self.depth = best_depth
        self.random_forest = RandomForestClassifier(random_state=0, max_depth= self.depth)
        self.train(data_train,labels) # train the model

    def predict(self, X):

        if len(X.shape) == 1:  # Predict on one sample
            class_label = np.array([self.random_forest.predict(X)])
            return class_label

        elif len(X.shape) == 2:  # Predict on multiple samples
            class_label = []
            for index in range(X.shape[0]):
                class_label.append(self.random_forest.predict(X[index].reshape(1,-1))[0])
            return class_label
