#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# By Pramod Sharma : pramod.sharma@prasami.com

# import statements

import os
import json

# Some global variables

JSON_DIR = '../json'

SETUP_JSON = 'setup.json'

class Config(object): # extend from class object
    
    def __init__(self):
        
        if os.path.exists(JSON_DIR):

            baseJsonPath = os.path.join(JSON_DIR, SETUP_JSON)

            if os.path.exists(baseJsonPath):

                with open(baseJsonPath) as f: # load the json file

                    setup_dict = json.load(f) 
        
        self._config = setup_dict # set it to conf

    def get_property(self, propertyName):
        
        if propertyName not in self._config.keys(): # we don't want KeyError
            
            return None  # just return None if not found
        
        return self._config[propertyName]
    
    def get_keys(self):
        
        return self._config.keys()
