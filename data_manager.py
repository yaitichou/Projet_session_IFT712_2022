# -*- coding: utf-8 -*-

#####
# - caiy2401 - CAI, Yunfan
# - gaye1902 - ElHadji Habib Gaye
# - aity1101 - AIT ICHOU, Yoann
###

import itertools
import numpy as np
from sklearn import preprocessing
import pandas as pd
import csv
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt




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
    labels_temp = []
    ids = []

    with open(path,'rt') as csv_file:

        # creating an object of csv reader
        # with the delimiter as ,
        csv_reader = csv.reader(csv_file, delimiter = ',')

        # loop to iterate through the rows of csv
        for row in csv_reader:
            if(dataset == "train"): # training dataset
                
                ids.append(row[0]) # get leaf id
                labels_temp.append(row[1]) # get species label (second column)
                data.append(row[2:]) # get data rows without id

            else: # testing dataset

                ids.append(row[0]) # get leaf id
                data.append(row[1:]) # get data rows without id


    attribute_names = data[0] # get the attributes names
    data = np.array(data[1:]) # delete attributes names from dataset
    labels = np.array(labels[1:])

    df = pd.DataFrame(data, columns = attribute_names) # get the attributes names

    return attribute_names, data, ids, labels


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

def to_array(data_df):
    """
    Method that change a dataframe into an array
    """
    data_array = data_df.to_numpy()
    return data_array

def generate_combinations_for_attributes(attributes_list, max_length):
    """
    Method that generate a list of attributes combinations

    Function found here : https://stackoverflow.com/questions/464864/how-to-get-all-possible-combinations-of-a-list-s-elements
    """
    generated_combinations = []

    for length_subset in range(np.minimum(len(attributes_list),max_length + 1)):
        for subset in itertools.combinations(attributes_list, length_subset):
            generated_combinations.append(subset)

    return generated_combinations

def ACP(data_matrix):
    """
    Method that applies the ACP and that plots the results of the ACP

    """
    n_components = 10 
    mypca = PCA(n_components=n_components)
    data = mypca.fit_transform(data_matrix)
    plt.figure()
    plt.title("histogramme de la répartition des données selon les "+str(n_components)+" composantes principales")
    plt.xlabel("composantes")
    plt.ylabel("variance")
    
    #we normalise std for explain that 2 first components are the most representative
    data = mypca.explained_variance_
    variance = []
    min_data = np.amin(data)
    for i in data:
        variance.append(i / min_data)
    value_variance_hist = []
    value_component = 1
    for i in variance:
        j = 0
        while j != int(i):
            value_variance_hist.append(value_component)
            j += 1
            
        value_component+=1
    label, counts = np.unique(value_variance_hist,return_counts=True)
    plt.bar(label,counts,align="center")
    
    #we realise the projection on the 2 first components
    mypca = PCA(n_components=2)

    
    data_fitted = mypca.fit_transform(data_matrix)
    print(mypca.components_)

    data_fitted_x = [element[0] for element in data_fitted]
    data_fitted_y = [element[1] for element in data_fitted]
    plt.figure()
    plt.scatter(data_fitted_x,data_fitted_y)
    
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Nuage de point issue d'une analyse en composantes principales")
    
    plt.figure()
    plt.title("Représentation de la corrélation entre les variables note d'un livre")
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.xlim([-0.5,1])
    plt.ylim([-1,1])

    #Call the function. 
    myplot(mypca.components_.T)
    
    plt.show()

#function obtained from stackoverflow : https://stackoverflow.com/questions/57340166/how-to-plot-the-pricipal-vectors-of-each-variable-after-performing-pca
def myplot(coeff,labels=None):
    """Function permitting to plot correlation of the different variables of a book

    Args:
        coeff (array): coeff of each vector
        labels (list, optional): labels for each vector Defaults to None.
    """
    n = coeff.shape[0]

    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
