# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division

import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.python.tools import freeze_graph

sys.path.append('../../')
from libs.configs import cfgs
from libs.networks import build_whole_network_NO_NMS_ARRAY

def build_detection_graph():
    # 1. preprocess img
    img_plac = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3],
                              name='input_img')  # is RGB. not GBR
    raw_shape = tf.shape(img_plac)
    raw_h, raw_w = tf.to_float(raw_shape[0]), tf.to_float(raw_shape[1])

    img_batch = tf.cast(img_plac, tf.float32)
    
    if cfgs.NET_NAME in ['resnet152_v1d', 'resnet101_v1d', 'resnet50_v1d']:
        img_batch = (img_batch / 255 - tf.constant(cfgs.PIXEL_MEAN_)) / tf.constant(cfgs.PIXEL_STD)
    else:
        img_batch = img_batch - tf.constant(cfgs.PIXEL_MEAN)
    
    img_batch = tf.expand_dims(img_batch, axis=0)  # [1, None, None, 3]

    det_net = build_whole_network_NO_NMS_ARRAY.DetectionNetwork(base_network_name=cfgs.NET_NAME,
                                                          is_training=False)

    detected_boxes, detection_scores, detection_category, final_allscores = det_net.build_whole_detection_network(
        input_img_batch=img_batch,
        gtboxes_batch=None,
        gtboxes_r_batch=None,gpu_id=0)
    

    x_c, y_c, w, h, theta = detected_boxes[:, 0], detected_boxes[:, 1], \
                            detected_boxes[:, 2], detected_boxes[:, 3], detected_boxes[:, 4]

    
    
    boxes = tf.transpose(tf.stack([x_c, y_c, w, h, theta]))
    dets = tf.concat([tf.reshape(detection_category, [-1, 1]),
                     tf.reshape(detection_scores, [-1, 1]),
                     boxes,final_allscores], axis=1, name='DetResults')

    return dets


def export_frozenPB(CKPT_PATH,OUT_DIR,PB_NAME):
    PB_NAME=PB_NAME+'.pb'
    
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    tf.reset_default_graph()

    dets = build_detection_graph()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        print("we have restred the weights from =====>>\n", CKPT_PATH)
        saver.restore(sess, CKPT_PATH)

        tf.train.write_graph(sess.graph_def, OUT_DIR, PB_NAME)
        freeze_graph.freeze_graph(input_graph=os.path.join(OUT_DIR, PB_NAME),
                                  input_saver='',
                                  input_binary=False,
                                  input_checkpoint=CKPT_PATH,
                                  output_node_names="DetResults",
                                  restore_op_name="save/restore_all",
                                  filename_tensor_name='save/Const:0',
                                  output_graph=os.path.join(OUT_DIR, PB_NAME.replace('.pb', '_Frozen.pb')),
                                  clear_devices=False,
                                  initializer_nodes='')


        os.remove(os.path.abspath(os.path.join(OUT_DIR,PB_NAME)))
        print('The model (frozen graph file) is at', os.path.abspath(os.path.join(OUT_DIR,PB_NAME.replace('.pb', '_Frozen.pb'))))