#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 15:37:38 2021

@author: francisco camargo
"""

import yaml

# https://stackoverflow.com/questions/1773805/how-can-i-parse-a-yaml-file-in-python

def read_yaml(filename):

    with open(filename, 'r') as stream:
        try:
            dictionary = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            
    return dictionary