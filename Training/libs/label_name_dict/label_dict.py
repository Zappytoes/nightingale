# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
from libs.configs import cfgs

####################################
##-------NAME_LABEL_MAP block------#
####################################
if cfgs.DATASET_NAME == 'OMITTED':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'class1': 1,
        'class2': 2,
        'class3': 3
    }

    # Set the training NMS threshold for each class
    threshold = {'class1':0.3,'class2':0.3,'class3':0.3} # these are hyperparameters you can play with

else:
    assert 'please set label dict!'
####################################
##----END NAME_LABEL_MAP block-----#
####################################


def get_label_name_map():
    reverse_dict = {}
    for name, label in NAME_LABEL_MAP.items():
        reverse_dict[label] = name
    return reverse_dict


LABEL_NAME_MAP = get_label_name_map()