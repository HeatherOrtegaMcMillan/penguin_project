################################# Prep Penguins Data #################################
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np 


def prep_penguins(df):
    # drop island and sex columns
    penguins_clean = df.drop(columns=['island', 'sex'])
    
    # create dictionary of old column names, new names removed the measurements
    col_names = {'bill_length_mm': 'bill_length', 
                 'bill_depth_mm': 'bill_depth', 
                 'flipper_length_mm': 'flipper_length', 
                 'body_mass_g': 'body_mass'}

    # rename columns
    penguins_clean = penguins_clean.rename(columns=col_names)
    
    # drop the 2 rows that had null values (3 and 339)
    penguins_clean = penguins_clean.dropna(axis=0)
    
    # split data using sklearn train test split
    # 20% test 80% train_validate (70% train, 30% validate)
    train, test = train_test_split(penguins_clean, test_size=0.2, random_state=713, stratify=penguins_clean.species)
    train, validate = train_test_split(train, train_size=0.7, random_state=713, stratify=train.species)
    
    return train, validate, test