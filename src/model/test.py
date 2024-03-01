#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 16:00:24 2021

@author: francisco camargo
"""

from sklearn.metrics import classification_report

def test(df, model_object):

    y_Test = df['y']
    X_Test = df.drop(columns=['y'])

    # Inference
    predictions = model_object.predict(X_Test) 
    print('\nReport')
    print(classification_report(y_Test, predictions, digits=3))