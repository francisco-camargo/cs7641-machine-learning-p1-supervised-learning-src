#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 15:56:44 2021

@author: francisco camargo
"""

import pandas as pd
import matplotlib.pyplot as plt
plt.close('all')
import numpy as np # need this to run eval()
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

try:
    import sys
    if '..' not in sys.path:
        sys.path.append('..')
    import src.read_yaml as ry
    from src.visualization.plot_learning_curve import plot_learning_curve 
    from src.visualization.plot_validation_curve import plot_validation_curve
except:
    import read_yaml as ry
    from visualization.plot_learning_curve import plot_learning_curve 
    from visualization.plot_validation_curve import plot_validation_curve


def pipeline_helper(dict_in, prefix):
    # return {'class__'+key:val for key, val in dict_in.items()}
    return {prefix +'__' + key:val for key, val in dict_in.items()}


def param_grid_helper(my_dict):
    # Construct parameters over which we perform GridSearchCV()
    param_grid = {}
    for key, val in my_dict.items():
        try:
            param_grid[key] = eval(val) # https://stackoverflow.com/questions/701802/how-do-i-execute-a-string-containing-python-code-in-python
        except:
            param_grid[key] = val
    return param_grid


def shufflesplit_helper(n_splits, test_size=0.2, random_state=42):
    return ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)


def train(df, project_alias, model_choice, train_dict):

    y_TV = df['y']
    X_TV = df.drop(columns=['y'])

    if model_choice == 'DT': # Decision Tree
        learner = DecisionTreeClassifier()
    elif model_choice == 'ADA':
        # learner = AdaBoostClassifier() # even though DTClassifier is default, this won't work when it's time to set value of hyperparameters
        learner = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
    elif model_choice == 'SVM': # Support Vector Machine
        learner = SVC()
    elif model_choice == 'KNN': # k-Nearest Neighbors
        learner = KNeighborsClassifier()
    elif model_choice == 'MLP':
        learner = MLPClassifier()
    try:
        learner.set_params(random_state=train_dict['random_state'])
    except:
        pass

    pipeline = Pipeline([ ('standardscaler', StandardScaler()) ])

    # Oversampling for class-imbalance
        # https://towardsdatascience.com/imbalanced-class-sizes-and-classification-models-a-cautionary-tale-part-2-cf371500d1b3
        # https://stackoverflow.com/questions/50245684/using-smote-with-gridsearchcv-in-scikit-learn
    if train_dict['oversample']:
        print('Will Oversample!!!')
        pipeline.steps.append(('sampling', SMOTE(random_state=train_dict['random_state'])))
    pipeline.steps.append(('class', learner))

    # choose type of experiment to perform
    # GridSearchCV
    if train_dict['train_mode'] == 'best':
        # Set hard-coded parameters for this experiment
        fixed_params = train_dict[model_choice]['fixed_param']
        pipeline.set_params(**pipeline_helper(fixed_params, 'class'))
        pipeline.fit(X=X_TV, y=y_TV)
        # from sklearn import tree
        # tree.plot_tree(pipeline['class'])
        return pipeline

    elif train_dict['train_mode'] == 'grid':

        param_grid = param_grid_helper(train_dict[model_choice]['gridsearch']['param_grid'])
        param_grid = pipeline_helper(param_grid, 'class')

        # Set hard-coded parameters for this experiment
        fixed_param = train_dict[model_choice]['fixed_param']
        pipeline.set_params(**pipeline_helper(fixed_param, 'class'))

        # Fit
        print('\tGridSearch verbose:')
        grid = GridSearchCV(
            pipeline,
            param_grid,
            scoring = train_dict['scoring'],
            n_jobs  = train_dict['n_jobs'],
            refit   = True,
            cv      = train_dict[model_choice]['gridsearch']['grid_cv'],
            verbose = train_dict['verbose'],
            return_train_score = train_dict['return_train_score'],
            )
        grid.fit(X=X_TV, y=y_TV)

        # Best params
        print('\tBest hyperparameters:', grid.best_params_)
        print('\tBest score:', round(grid.best_score_*100,2))

        return grid, pd.DataFrame(grid.cv_results_)

    # Validation Curve
    elif train_dict['train_mode'] == 'vali':
        experiment = train_dict['validation_exp']

        # Set hard-coded parameters for this experiment
        fixed_param = train_dict[model_choice]['fixed_param']
        pipeline.set_params(**pipeline_helper(fixed_param, 'class'))

        # Define independant variable for this experiment
        param_dict  = param_grid_helper(train_dict[model_choice][experiment]['ind_var'])
        param_name  = 'class__' + list(param_dict.items())[0][0]
        param_range = list(param_dict.items())[0][1]

        # cv = ShuffleSplit(**train_dict['shufflesplit'])
        cv = shufflesplit_helper(n_splits=train_dict[model_choice][experiment]['n_splits'], random_state=train_dict['random_state'])

        plot_validation_curve(estimator=pipeline,
                            project_alias = project_alias,
                            model_choice = model_choice,
                            experiment = experiment,
                            X=X_TV,
                            y=y_TV,
                            param_name=param_name,
                            param_range=param_range,
                            ylim=None,
                            xscale=train_dict[model_choice][experiment]['xscale'],
                            cv=cv,
                            scoring=train_dict['scoring'],
                            n_jobs=train_dict['n_jobs'])

    # Learning Curve
    elif train_dict['train_mode'] == 'learn':
        # Set hard-coded parameters for this experiment
        fixed_param = train_dict[model_choice]['fixed_param']
        pipeline.set_params(**pipeline_helper(fixed_param, 'class'))

        cv = shufflesplit_helper(n_splits=train_dict[model_choice]['learning']['n_splits'], random_state=train_dict['random_state'])

        try:
            train_sizes = eval(train_dict[model_choice]['learning']['train_sizes'])
        except:
            train_sizes = train_dict[model_choice]['learning']['train_sizes']

        plot_learning_curve(estimator=pipeline,
                            project_alias = project_alias,
                            model_choice = model_choice,
                            X=X_TV,
                            y=y_TV,
                            axes=None,
                            ylim=None,
                            xscale=train_dict[model_choice]['learning']['xscale'],
                            cv=cv,
                            scoring=train_dict['scoring'],
                            n_jobs=train_dict['n_jobs'],
                            train_sizes=train_sizes)


def main():
    config_list = ['../../configs/config_car.yaml', '../../configs/config_monk.yaml', '../../configs/config_cancer.yaml']
    config_list = ['../../configs/config_monk.yaml']
    # config_list = ['../../configs/config_cancer.yaml']
    
    for config_file in config_list:
        config_dict = ry.read_yaml(config_file)

        train_pkl = config_dict['preprocess']['interim_dir']+config_dict['project_alias']+'_interim_train.pkl'
        try:
            df = pd.read_pickle(train_pkl)
        except:
            df = pd.read_pickle('../../'+train_pkl)

        for model_choice in config_dict['model_choice']:
            print('\nTrain ' + config_dict['project_alias'] + ' with ' + model_choice)
            results = train(df, config_dict['project_alias'], model_choice, train_dict=config_dict['train'])
    return results


if __name__ == "__main__":

    results = main()