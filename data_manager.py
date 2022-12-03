# -*- coding: utf-8 -*-

#####
# - caiy2401 - CAI, Yunfan
# - gaye1902 - ElHadji Habib Gaye
# - aity1101 - AIT ICHOU, Yoann
###

import numpy as np
import csv




def data_extract(path, dataset = 'train'):
    """
    Method that extracts the data used to train and test
    our different models
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
    data = data[1:] # delete attributes names from dataset

    return attribute_names, data, ids, labels

def data_cleaning(data):
    """
    Method that extracts the data used to train and test
    our different models
    """
    
    cleaned_data = data ## To remove


    return cleaned_data
