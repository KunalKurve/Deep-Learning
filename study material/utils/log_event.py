#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# By Pramod Sharma : pramod.sharma@prasami.com

'''
    This file logs events
    To be included in all files
'''

# Import libraries
from __future__ import division, print_function, absolute_import
import os
from datetime import datetime
import logging
from utils.configuration import Config
  
def fn_log_event ( event_text, event_type ) :
    
    config = Config()
    
    logDir = config.get_property('_LOG_DIR')

    logFilename = 'events_log_{}.txt'.format(datetime.now().strftime("%Y_%m_%d"))
    
    logFilePath = os.path.join(logDir, logFilename)

    logging.basicConfig(format='%(asctime)s : %(levelname)s:%(message)s',
                        datefmt='%Y-%m-%d %I:%M:%S %p',
                        filename=logFilePath,
                        level=logging.DEBUG)

    if event_type =='debug':
        
        logging.debug(event_text)

    if event_type =='info':
        
        logging.info(event_text)

    if event_type =='warning':
        
        logging.warning(event_text)

#if __name__ == '__main__':

    #fn_log_event('This is debug info from py file', 'debug')
    #fn_log_event('This is info only from py file', 'info')
    #fn_log_event('This is a warning from py file', 'warning')
