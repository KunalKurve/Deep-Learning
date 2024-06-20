#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# By Pramod Sharma : pramod.sharma@prasami.com

# Import Statements

#---------------------------
import os
import sys
import json
import glob
import platform

import time
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from PIL.ExifTags import TAGS
import piexif


import logging
from util.log_event import fn_log_event

from util.configuration import Config

config = Config()

ratioImage = config.get_property('RATIO_IMAGE')

def fn_dir_exists(_dir):
    '''
        verify if the directory exists
    '''

    res = os.path.exists(_dir)

    fn_log_event ('-- Directory "{}" exist : {}'.format(_dir, res), 'debug')

    if not res:

        fn_log_event ('-- Directory "{}" does not exists.'.format(_dir), 'debug')

        sys.exit('-- Directory "{}" does not exists.'.format(_dir))

    return res


def fn_file_exists(_file):
    '''
        verify if the file exists
    '''
    res = os.path.exists(_file)

    fn_log_event ('-- File "{}" exist : {}'.format(_file, res), 'debug')

    if not res:

        fn_log_event ('-- File "{}" does not exists.'.format(_file), 'debug')

        sys.exit('-- File "{}" does not exists.'.format(_file))

    return res

### following two routines to add image taken time to the images
def fn_get_date_taken(path):

    return Image.open(path)._getexif()[36867]


def fn_add_creation_date(ts, destPath):

    exif_dict = {'Exif': { piexif.ExifIFD.DateTimeOriginal: ts    #.strftime("%Y:%m:%d %H:%M:%S")
                         }}

    exif_bytes = piexif.dump(exif_dict)

    piexif.insert(exif_bytes, destPath)


def get_image_data(imagePath, **kwargs):

    '''
        Takes path of the image and return the image
    '''

    img = Image.open(imagePath)

    assert img is not None, "Failed to read image : %s, %s" % (image_id)

    return img


def fn_save_image_data (img, fPath):
    '''
        takes image and path and saves the image
    '''

    img.save(fPath, "JPEG")



def fn_draw_rectangle(img,slotID, xMin, yMin, w, h, free = 1, use_normalized_coordinates = False):

    '''
        Function to mark the slots and color them as per occupancy
        Args:
            img    : the image object,
            slotID : Name of the slot,
            xMin   : upper x coordinate,
            yMin   : upper y coordinate,
            w      : width of the slot,
            h      : height,
            free   : free => 1 = free, 0 = occupied,
            use_normalized_coordinates : use it if the coordinates are ratios
    '''

    if free == 0 :

        lineColor = (255,0,0)

    if free == 1 :

        lineColor = (0,255,0)

    lineWidth = img.size[0] // ratioImage   # set line width as per the size of image


    displayStr = '{}'.format (slotID)

    draw = ImageDraw.Draw(img)

    im_width, im_height = img.size

    if use_normalized_coordinates:

        (left, right, top, bottom) = (xMin * im_width, (xMin + w) * im_width,

                                      yMin * im_height, (yMin + h) * im_height)

    else:

        (left, right, top, bottom) = (xMin, (xMin + w), yMin,  (yMin + h))



    draw.line([(left, top), (left, bottom), (right, bottom),

             (right, top), (left, top)], width=lineWidth, fill=lineColor)

    #  Selecting the fonts

    try:

        if platform.dist()[0] == 'Ubuntu':

            fontFace = '/usr/share/fonts/truetype/ubuntu-font-family/Ubuntu-B.ttf'

        else:

            fontFace = 'arial.ttf'

        font = ImageFont.truetype(fontFace, lineWidth * 5) # font size in proportion to line width

    except IOError:

        font = ImageFont.load_default()

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    #display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]

    display_str_heights = font.getsize(displayStr)[1]

    # Each display_str has a top and bottom margin of 0.05x.

    total_display_str_height = (1 + 2 * 0.05) * display_str_heights

    if top > total_display_str_height:

        text_bottom = top

    else:

        text_bottom = bottom + total_display_str_height

    text_width, text_height = font.getsize(displayStr)

    margin = np.ceil(0.05 * text_height)

    draw.rectangle([(left, text_bottom - text_height - 2 * margin),

                    (left + text_width + 2 * margin,text_bottom)], fill=lineColor)

    draw.text((left + margin, text_bottom - text_height - margin),
              displayStr, fill='black', font=font)

    text_bottom -= text_height - 2 * margin

    return img

def fn_draw_point(img, xCG, yCG, pt_type = 'car'):
    '''
        Function to mark CG of the Car
        Args:
            img : the image object,
            xCG : x coordinate of CG,
            yCG : y coordinate of CG,
            pt_type = 'car'
    '''

    if pt_type == 'car':

        fill="orange"

        outline = "Blue"

    else:

        fill="yellow"

        outline = "green"



    width = img.size[0] // ratioImage   # CG marker as per size of the image

    dr = ImageDraw.Draw(img)

    dr.rectangle(((xCG - 2* width, yCG - 2 * width),
                  (xCG + 2 * width, yCG + 2 * width )), fill=fill, outline = outline)

    return img

def fn_convert_timestamp(tstr):
    '''
        Function to conver string of form "2015-11-12 1444"
    '''
    
    return datetime.strptime(tstr, "%Y-%m-%d_%H%M")
