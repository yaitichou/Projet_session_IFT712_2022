import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score as ac
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report


class DTSClassifier(object):
    def __init__(self, depth = 1,  min_samples_split = 2, min_samples_leaf = 1, criterion = "gini"):
        self.depth = depth
        self.decision_tree = DecisionTreeClassifier(random_state=0,criterion=criterion, max_depth=depth, min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf)


    def train(self,data_train,labels):
        self.decision_tree = self.decision_tree.fit(data_train, labels)
        
    
    def cross_validation(self,data_train,labels,depth_list,split_list,criterion_list,min_samples_leaf_list):
        grilleRecherche = list()
        
        for grid_min_sample_leaf in min_samples_leaf_list:
            for grid_criterion in criterion_list:
                for grid_split in split_list:
                    for grid_depth in depth_list:
                        grilleRecherche.append({"depth": grid_depth, "min_samples_split": grid_split, "criterion":grid_criterion,"min_samples_leaf":grid_min_sample_leaf})
        
        k = 5

        kf = KFold(n_splits=k, random_state=None, shuffle=True)
        accuracy_hyperparameter = {}
        precision_hyperparameter = {}
        f1_hyperparameter = {}
        for hyper in range(len(grilleRecherche)):

            min_samples_leaf = grilleRecherche[hyper]["min_samples_leaf"]
            criterion = grilleRecherche[hyper]["criterion"]
            depth = grilleRecherche[hyper]["depth"]
            min_samples_split = grilleRecherche[hyper]["min_samples_split"]

            decision_tree = DTSClassifier(depth = depth, min_samples_split=min_samples_split, criterion=criterion,min_samples_leaf=min_samples_leaf)
            accuracy = 0
            precision = 0
            f1 = 0
            for train_index, validation_index in kf.split(labels):

                X_train, X_validation = data_train[train_index], data_train[validation_index]
                y_train, y_validation = labels[train_index], labels[validation_index]

                decision_tree.train(X_train,y_train) # train the model

                predictions = decision_tree.predict(X_validation)
                accuracy += ac(y_validation,predictions) # compute the accuracy
                precision += precision_score(y_validation,predictions, average = "micro") # compute the precision
                f1 += f1_score(y_validation,predictions, average = "micro") # compute the precision
            
            accuracy_hyperparameter[hyper] = accuracy / k
            precision_hyperparameter[hyper] = precision / k
            f1_hyperparameter[hyper] = f1 / k
            
        hyper = max(accuracy_hyperparameter, key=accuracy_hyperparameter.get)
        print("Best hyperparameters:", grilleRecherche[hyper])

        self.decision_tree = DecisionTreeClassifier(random_state=0, max_depth=grilleRecherche[hyper]["depth"],min_samples_split =grilleRecherche[hyper]["min_samples_split"],criterion=grilleRecherche[hyper]["criterion"],min_samples_leaf=grilleRecherche[hyper]["min_samples_leaf"])
        self.decision_tree = self.decision_tree.fit(data_train, labels)

        return accuracy_hyperparameter[hyper], f1_hyperparameter[hyper], precision_hyperparameter[hyper]


    def predict(self, X):

        if len(X.shape) == 1:  # Predict on one sample
            class_label = np.array([self.decision_tree.predict(X)])
            return class_label

        elif len(X.shape) == 2:  # Predict on multiple samples
            class_label = []
            for index in range(X.shape[0]):
                class_label.append(self.decision_tree.predict(X[index].reshape(1,-1))[0])
            return class_label

        

