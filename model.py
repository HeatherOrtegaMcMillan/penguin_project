############ Modeling Functions for In Vino Veritas ##################
# This module has Functions pertaining to feature selection, creating models,
# and evaluating models

import pandas as pd 
import numpy as np 
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, confusion_matrix, recall_score, plot_confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns


###############
def select_kbest(X, y, k, score_func=f_classif):
    '''
    takes in the predictors (X), the target (y), and the number of features to select (k) 
    and returns the names (in a list) of the top k selected features based on the SelectKBest class
    Optional arg: score_func. Default is f_regression. other options ex: f_classif 
    '''
    # create selector
    f_selector = SelectKBest(score_func=score_func, k=k)
    
    #fit to X and y
    f_selector.fit(X, y)
    
    # return the list of the column names that are the top k selected features
    return list(X.columns[f_selector.get_support()])

###############
def all_aboard_the_X_train(X_cols, y_col, train, validate, test):
    '''
    X_cols = list of column names you want as your features
    y_col = string that is the name of your target column
    train = the name of your train dataframe
    validate = the name of your validate dataframe
    test = the name of your test dataframe
    outputs X_train and y_train, X_validate and y_validate, and X_test and y_test
    6 variables come out! So have that ready
    '''
    
    # do the capital X lowercase y thing for train test and split
    # X is the data frame of the features, y is a series of the target
    X_train, y_train = train[X_cols], train[y_col]
    X_validate, y_validate = validate[X_cols], validate[y_col]
    X_test, y_test = test[X_cols], test[y_col]
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test

##############

def print_metrics(model, X, y, pred, set_name = 'This Set', class_names = None):

    '''
    This function prints out a classification report and confusion matrix
    in the form of a heatmap for a given model.

    Optional argument class_names
        Default is None. This will automatically name the classes from 1-n
        class_names = 'auto' option if you want it to figure out the class names for you
        class_names = list of aliases

    ex: print_metrics(cls, X_train, y_train, train_pred, set_name = 'Train')
    '''
    # print out model you're using including hyper parameters
    print(model)
    # Set name (train validate or test)
    print(f"~~~~~~~~{set_name} Scores~~~~~~~~~")
    # print classification report using predictions
    print(classification_report(y, pred))
    
    # if no class name
    if class_names == None:
        # get number of unique classes in the y dataset
        classes = y.nunique()
        # make list of 1 to however many classes there are, as strings
        class_names = list(np.arange(1, classes+1).astype(str)) 

    # figure out class names for you
    if class_names == 'auto':
        # get the unique values in the y
        class_names = y.unique()

    # set up lables for the classes (specific to this data)
    class_names = np.array(class_names)

    # create custom maroon color map
    maroon_cmap = sns.color_palette("light:#D0894B", as_cmap=True)

    # graph confusion matrix using heatmap
    with sns.axes_style("white"):
        matrix = plot_confusion_matrix(model,X, y, display_labels=class_names, 
                                        cmap = maroon_cmap)
        plt.grid(False)
        plt.show()
        print()

