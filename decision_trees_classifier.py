


import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score as ac
from sklearn.model_selection import KFold 

from sklearn.tree import DecisionTreeClassifier



class DTSClassifier(object):


    def __init__(self, depth = 1, criterion = "gini"):
        self.depth = depth
        self.decision_tree = DecisionTreeClassifier(random_state=0,criterion=criterion, max_depth=depth)


    def train(self,data_train,labels):
        self.decision_tree = self.decision_tree.fit(data_train, labels)
        
    def plot_tree(self):
        tree.plot_tree(self.decision_tree)
    
    def cross_validation(self,data_train,labels,depth_max):

        accuracy_hyperparameter = np.zeros(depth_max) 
        k = 10

        kf = KFold(n_splits=k, random_state=None, shuffle=True)

        for depth in range(1,depth_max):
            decision_tree_classifier = DTSClassifier(depth)
            accuracy = 0
            for train_index, validation_index in kf.split(labels):

                X_train, X_validation = data_train[train_index], data_train[validation_index]
                y_train, y_validation = labels[train_index], labels[validation_index]

                decision_tree_classifier.train(X_train,y_train) # train the model

                predictions = decision_tree_classifier.predict(X_validation)
                accuracy += ac(y_validation,predictions) # compute the accuracy

            accuracy_hyperparameter[depth-1] = accuracy / k
            
        best_depth = np.argmax(accuracy_hyperparameter) + 1

        print("best_depth >> "+str(best_depth )+" accuracy >> "+str(accuracy_hyperparameter[best_depth - 1]))

        self.depth = best_depth
        self.decision_tree = DecisionTreeClassifier(random_state=0, max_depth= self.depth)
        self.train(data_train,labels) # train the model

    def predict(self, X):

        if len(X.shape) == 1:  # Predict on one sample
            class_label = np.array([self.decision_tree.predict(X)])
            return class_label

        elif len(X.shape) == 2:  # Predict on multiple samples
            class_label = []
            for index in range(X.shape[0]):
                class_label.append(self.decision_tree.predict(X[index].reshape(1,-1))[0])
            return class_label
    