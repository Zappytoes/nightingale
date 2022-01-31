# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf

"""
Nightingale config file used for pre-processing and training SCRDet++

Modify the first block of definitions to match your data, environment, and experiment

"""
###########################################################################
# ------------- Basic Settings --------------------------------------------
# ------------- MODIFY this block of code ---------------------------------
###########################################################################
VERSION = 'MyCoolExperiment' # this will be the name of the folder that gets created under 
                             # Training/output directory for your checkpoint & event files

NET_NAME = 'resnet152_v1d'  # must match one of the network names available in 
                            # the Training/data/pretrained_weights directory. 
                            # Currentlly suppoted networks are resnet_v1_50,
                            # resnet101_v1d, and resnet152_v1d. 

DATASET_NAME = 'OMITTED'    # The name you want to give your dataset. 

ROOT_PATH = 'whatever/Nightingale/Training/' # The ROOT_PATH is the absolute path to the "Nightingale/Training" folder

#TFRECORD_PATH = os.path.join(ROOT_PATH,'data/tfrecord_omitted') # (see Notebook #2) the directory that contains 
                                                               # (or will contain) the tensorflow record file. 
#TFRECORD_NAME = DATASET_NAME+'_train.tfrecord' # # (see Notebook #2) the name of the Tensorflow Record composed of your training data

CLASS_NUM = 3 # number of classes in your dataset, not including background. 

GPU_GROUP = "0,1,2,3" # a list of GPU's to use for training

SAVE_WEIGHTS_INTE = 1000 # the frequency at which to write tensorflow checkpoint files. Also affects the 
                          # learning rate decay and how long to train for (see vars DECAY_STEP & MAX_ITERATION)

TRAIN_IMG_CHIP_SIZE = 2000 # Recommended pixel width/height setting for chiping NITF imagery for training. This 
                           # is approximately the max image size that can be processed during training with 
                           # resnet152_v1d on Tesla v100 SXM2 GPU's with ~34Gb of memory. 

TRAIN_IMG_CHIP_OVRLP = 200 # NITF image chip overlap for training. Consider increasing this 
                           # if you suspect objects are large relative to the chip size (overlaping
                           # images should allow each object to be fully present, i.e., not cropped, 
                           # on at least one of the overlapping training images). 
############################################################################
# -------------------- END Basic Settings ----------------------------------
# -------------------- END Modifications -----------------------------------
############################################################################
#
TFRECORD_PATH = os.path.join(ROOT_PATH,'data/tfrecord_'+DATASET_NAME) ##Do not change. This is the directory that contains 
                                                               # (or will contain) the tensorflow record file.
TFRECORD_NAME = DATASET_NAME+'_train.tfrecord' #Do not change. This is the name of the Tensorflow Record composed of your training data
#
# ------ Advanced settings below ------------
# ---------------------------------------- System_config
ADD_BOX_IN_TENSORBOARD = True
ROTATE_NMS_USE_GPU = True
print (20*"++--")
print ('The ROOT_PATH is ',ROOT_PATH)
NUM_GPU = len(GPU_GROUP.strip().split(','))
SHOW_TRAIN_INFO_INTE = 50 # frequency to print training info
SMRY_ITER = 500 # for tensorboard summary


SUMMARY_PATH = ROOT_PATH +'/output/summary'
TEST_SAVE_PATH = ROOT_PATH + '/tools/test_result'

if NET_NAME.startswith("resnet"):
    weights_name = NET_NAME
elif NET_NAME.startswith("MobilenetV2"):
    weights_name = "mobilenet/mobilenet_v2_1.0_224"
else:
    raise Exception('net name must in [resnet_v1_101, resnet_v1_50, MobilenetV2]')

PRETRAINED_CKPT = ROOT_PATH + '/data/pretrained_weights/' + weights_name + '.ckpt'
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')
EVALUATE_DIR = ROOT_PATH + '/output/evaluate_result_pickle/'

# ------------------------------------------ Train config
RESTORE_FROM_RPN = False
IS_FILTER_OUTSIDE_BOXES = False
FREEZE_BLOCKS = [True, True, False, False, False]  # for gluoncv backbone
FIXED_BLOCKS = 0  # allow 0~3
USE_07_METRIC = True
CUDA9 = True

RPN_LOCATION_LOSS_WEIGHT = 1.
RPN_CLASSIFICATION_LOSS_WEIGHT = 1.0
FAST_RCNN_LOCATION_LOSS_WEIGHT = 1.0
FAST_RCNN_CLASSIFICATION_LOSS_WEIGHT = 1.0
RPN_SIGMA = 3.0
FASTRCNN_SIGMA = 1.0

MUTILPY_BIAS_GRADIENT = 2.0  # if None, will not multipy
GRADIENT_CLIPPING_BY_NORM = 10.0  # if None, will not clip

BATCH_SIZE = 1
EPSILON = 1e-5
MOMENTUM = 0.9
LR = 0.001 * BATCH_SIZE * NUM_GPU
DECAY_STEP = [SAVE_WEIGHTS_INTE*12, SAVE_WEIGHTS_INTE*16, SAVE_WEIGHTS_INTE*20]
MAX_ITERATION = SAVE_WEIGHTS_INTE*25
WARM_SETP = int(1.0 / 4.0 * SAVE_WEIGHTS_INTE)

# -------------------------------------------- Data_preprocess_config
PIXEL_MEAN = [108.108, 108.108, 108.108]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
PIXEL_MEAN_ = [0.424, 0.424, 0.424] #used for v1d 
PIXEL_STD = [0.23, 0.23, 0.23]

IMG_SHORT_SIDE_LEN = [TRAIN_IMG_CHIP_SIZE, TRAIN_IMG_CHIP_SIZE, TRAIN_IMG_CHIP_SIZE-250, TRAIN_IMG_CHIP_SIZE-500]
IMG_MAX_LENGTH = TRAIN_IMG_CHIP_SIZE


IMG_ROTATE = True
RGB2GRAY = False
VERTICAL_FLIP = True
HORIZONTAL_FLIP = True
IMAGE_PYRAMID = True

# --------------------------------------------- Network_config
INITIALIZER = tf.random_normal_initializer(mean=0.0, stddev=0.01)
BBOX_INITIALIZER = tf.random_normal_initializer(mean=0.0, stddev=0.001)
WEIGHT_DECAY = 0.00004 if NET_NAME.startswith('Mobilenet') else 0.0001

# ---------------------------------------------Anchor config
USE_CENTER_OFFSET = False

LEVLES = ['P2', 'P3', 'P4', 'P5', 'P6']
BASE_ANCHOR_SIZE_LIST = [32, 64, 128, 256, 512]  # addjust the base anchor size for voc.
ANCHOR_STRIDE_LIST = [4, 8, 16, 32, 64]
ANCHOR_SCALES = [1.0]
ANCHOR_RATIOS = [0.5, 1., 2.0, 1/4.0, 4.0, 1/6.0, 6.0]
ROI_SCALE_FACTORS = [10., 10., 5.0, 5.0, 2.0]
ANCHOR_SCALE_FACTORS = None

# --------------------------------------------FPN config
SHARE_HEADS = True
KERNEL_SIZE = 3
RPN_IOU_POSITIVE_THRESHOLD = 0.7
RPN_IOU_NEGATIVE_THRESHOLD = 0.3
TRAIN_RPN_CLOOBER_POSITIVES = False

RPN_MINIBATCH_SIZE = 512  # 256
RPN_POSITIVE_RATE = 0.5
RPN_NMS_IOU_THRESHOLD = 0.7  # 0.7
RPN_TOP_K_NMS_TRAIN = 12000
RPN_MAXIMUM_PROPOSAL_TARIN = 2000

RPN_TOP_K_NMS_TEST = 6000
RPN_MAXIMUM_PROPOSAL_TEST = 1000

# -------------------------------------------Fast-RCNN config
ROI_SIZE = 14
ROI_POOL_KERNEL_SIZE = 2
USE_DROPOUT = False
KEEP_PROB = 1.0
SHOW_SCORE_THRSHOLD = 0.6  # only show in tensorboard

SOFT_NMS = False
FAST_RCNN_NMS_IOU_THRESHOLD = 0.3  # 0.5
FAST_RCNN_NMS_MAX_BOXES_PER_CLASS = 200
FAST_RCNN_IOU_POSITIVE_THRESHOLD = 0.5
FAST_RCNN_IOU_NEGATIVE_THRESHOLD = 0.0   # 0.1 < IOU < 0.5 is negative
FAST_RCNN_MINIBATCH_SIZE = 512  # if is -1, that is train with OHEM
FAST_RCNN_POSITIVE_RATE = 0.25

ADD_GTBOXES_TO_TRAIN = False

# -------------------------------------------mask config
USE_SUPERVISED_MASK = True
MASK_TYPE = 'r'  # r or h
BINARY_MASK = False
SIGMOID_ON_DOT = False
MASK_ACT_FET = True  # weather use mask generate 256 channels to dot feat.
GENERATE_MASK_LIST = ["P2", "P3", "P4", "P5"]
ADDITION_LAYERS = [4, 4, 4, 4]  # add 4 layer to generate P2_mask, 2 layer to generate P3_mask
ENLAEGE_RF_LIST = ["P2", "P3", "P4", "P5"]
SUPERVISED_MASK_LOSS_WEIGHT = 0.1

# -------------------------------------------Tricks config
USE_CONCAT = True
CONCAT_CHANNEL = 1024  # 256

ADD_GLOBAL_CTX = True
ADD_EXTR_CONVS_FOR_REG = 8  # use 0 to do not use any extra convs



