################################# Prep Penguins Data #################################
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np 


# clean and prep function
def prep_penguins(df, explore = True):
    '''
    This function takes in the penguins dataframe and preps it
    Argument explore is set to True will not encode anything only split into train and test
    If explore is set to False, it will drop the island column and encode the sex column
    Renames columns to be more readable, also drops nulls 
    RETURNS: train, validate, test
    '''

    if explore == False:
        # drop island and sex columns
        df = df.drop(columns='island')
        # get dummy columns for sex category
        df = pd.get_dummies(df, columns = ['sex'], drop_first='False')
        df = df.rename(columns = {'sex_Male': 'is_male'})
    
    # create dictionary of old column names, new names removed the measurements
    col_names = {'bill_length_mm': 'bill_length', 
                 'bill_depth_mm': 'bill_depth', 
                 'flipper_length_mm': 'flipper_length', 
                 'body_mass_g': 'body_mass'
                }

    # rename columns
    df = df.rename(columns=col_names)
    
    # drop the 2 rows that had null values (3 and 339)
    df = df.dropna(axis=0)
    
    # split data using sklearn train test split
    # 20% test 80% train_validate (70% train, 30% validate)
    train, test = train_test_split(df, test_size=0.2, random_state=713, stratify=df.species)
    train, validate = train_test_split(train, train_size=0.7, random_state=713, stratify=train.species)
    
    return train, validate, test


def my_scaler(train, validate, test, col_names, scaler, scaler_name):
    
    '''
    This function takes in the train validate and test dataframes, columns you want to scale (as a list), a scaler (i.e. MinMaxScaler(), with whatever paramaters you need),
    scaler_name as a string.
    col_names: list of columns to scale
    Scaler_name, should be what you want in the name of your new dataframe columns.
    Adds columns to the train validate and test dataframes. 
    Outputs scaler for doing inverse transforms.
    Ouputs a list of the new column names (what you can use to create the X_train).
    
    example: min_max_scaler, scaled_cols_list = my_scaler(train, validate, test, MinMaxScaler(), 'scaled_min_max')
    
    '''
    
    #create the scaler (input here should be minmax scaler)
    mm_scaler = scaler
    
    # make empty list for return
    scaled_cols_list = []
    
    # loop through columns in col names
    for col in col_names:
        
        #fit and transform to train, add to new column on train df
        train[f'{col}_{scaler_name}'] = mm_scaler.fit_transform(train[[col]]) 
        
        #df['col'].values.reshape(-1, 1)
        
        #transform cols from validate and test (only fit on train)
        validate[f'{col}_{scaler_name}']= mm_scaler.transform(validate[[col]])
        test[f'{col}_{scaler_name}']= mm_scaler.transform(test[[col]])
        
        #add new column name to the list that will get returned
        scaled_cols_list.append(f'{col}_{scaler_name}')
    
    #confirmation print
    print('Your scaled columns have been added to your train validate and test dataframes.')
    
    #returns scaler, and a list of column names that can be used in X_train, X_validate and X_test.
    return scaler, scaled_cols_list 