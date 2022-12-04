# -*- coding: utf-8 -*-

#####
# - caiy2401 - CAI, Yunfan
# - gaye1902 - ElHadji Habib Gaye
# - aity1101 - AIT ICHOU, Yoann
###

import numpy as np
from sklearn import preprocessing
import pandas as pd
import csv




def data_extract(path, dataset = 'train'):
    """
    Method that extracts the data used to train and test
    our different models
    Inputs:
        - path: the path to the csv file
        - dataset: the kind of dataset ("train" or "test")
    Outputs:
        - attribute_names: the names of the attributes of the data, shape = (A)
        - data: the data of shape = (N,A) with attributes as columns and leafs as rows (without the labels and the ids)
        - ids: the ids of the leafs, shape = (N)
        - labels: the list of expected labels for the leafs, shape = (N)
    """


    # opening the csv file by specifying
    # the location
    # with the variable name as csv_file

    data = []
    labels = []
    ids = []

    with open(path,'rt') as csv_file:

        # creating an object of csv reader
        # with the delimiter as ,
        csv_reader = csv.reader(csv_file, delimiter = ',')

        # loop to iterate through the rows of csv
        for row in csv_reader:
            if(dataset == "train"): # training dataset
                
                ids.append(row[0]) # get leaf id
                labels.append(row[1]) # get species label (second column)
                data.append(row[2:]) # get data rows without id

            else: # testing dataset

                ids.append(row[0]) # get leaf id
                data.append(row[1:]) # get data rows without id


    attribute_names = data[0] # get the attributes names
    data = np.array(data[1:]) # delete attributes names from dataset
    
    df = pd.DataFrame(data, columns = attribute_names) # get the attributes names

    return attribute_names, df, ids, labels

def error(predictions,labels):
    """
    Method that computes the error of a model
    Inputs :
        - predictions : the predictions of the dataset, shape = (N)
        - labels : label of the data, shape = (N)
    Output :
        - error : the percentage of predictions that don't match the labels

    """

    return 100 * np.sum(np.abs(predictions - labels)) / len(labels)
    

def error(predictions,labels):
    """
    Method that computes the error of a model
    Inputs :
        - predictions : the predictions of the dataset, shape = (N)
        - labels : label of the data, shape = (N)
    Output :
        - error : the percentage of predictions that don't match the labels

    """

    return 100 * np.sum(np.abs(predictions - labels)) / len(labels)
    

def data_cleaning(data):
    """
    Method that extracts the data used to train and test
    our different models
    """
    
    scaler = preprocessing.StandardScaler().fit(data)
    cleaned_data = scaler.transform(data)

    return cleaned_data

