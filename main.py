#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 11:40:18 2021

@author: francisco camargo
"""

import argparse
import pandas as pd
import src.features.make_features as mf
import src.read_yaml as ry
import src.model.train as train
import src.model.test as test

def main(config_file):

    if not config_file:
        config_file = 'configs/config_monk.yaml'
        # config_file = 'configs/config_cancer.yaml'

    # Read .yaml config file
    config_dict = ry.read_yaml(config_file)
    project_alias = config_dict['project_alias']
    ml_steps = config_dict['ml_steps']
    
    # Pre-Process raw data
    if 'preprocess' in ml_steps:
        df_train, df_test = mf.make_features(project_alias, preprocess_dict=config_dict['preprocess'])

    # Train model
    if 'train' in ml_steps:
        if 'preprocess' not in ml_steps:
            train_pkl = config_dict['preprocess']['interim_dir']+project_alias+'_interim_train.pkl'
            test_pkl  = config_dict['preprocess']['interim_dir']+project_alias+'_interim_test.pkl'
            df_train, df_test = pd.read_pickle(train_pkl), pd.read_pickle(test_pkl)
        for model_choice in config_dict['model_choice']:
            print('\nTrain ' + project_alias + ' with ' + model_choice)
            model_object = train.train(df_train, project_alias, model_choice, train_dict=config_dict['train'])

    # Test model
    if 'test' in ml_steps:
        if 'train' not in ml_steps:
            model_object = 1
        test.test(df_test, model_object)

    # Plot results
    if 'plot' in ml_steps:
        pass

if __name__ == "__main__":

    # Command-Line Interface
        # $ python main.py -c configs/config_mushroom.yaml
        # https://pymotw.com/2/argparse/        
    parser = argparse.ArgumentParser(description='CS7641 Machine Learning: Assignment 1')
    parser.add_argument('-c', type=str, help='Provide path for .yaml config file to be used.')
    args = parser.parse_args()

    main(args.c)