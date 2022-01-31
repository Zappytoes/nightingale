# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from libs.configs import cfgs
import numpy as np
import numpy.random as npr
from libs.box_utils.cython_utils.cython_bbox import bbox_overlaps
from libs.box_utils import encode_and_decode

#import logging
#logging.basicConfig(filename='XXmulti_gpu_trainXX.log',level=logging.INFO)
#logger = logging.getLogger(__name__)


def anchor_target_layer(
        gt_boxes, img_shape, all_anchors, is_restrict_bg=False):
    """Same as the anchor target layer in original Fast/er RCNN """

    if len(gt_boxes) > 0:
    
        total_anchors = all_anchors.shape[0]
        img_h, img_w = img_shape[1], img_shape[2]
        gt_boxes = gt_boxes[:, :-1]  # remove class label


        # allow boxes to sit over the edge by a small amount
        _allowed_border = 0

        # only keep anchors inside the image
        if cfgs.IS_FILTER_OUTSIDE_BOXES: # this is set to false in current cfgs
            inds_inside = np.where(
                (all_anchors[:, 0] >= -_allowed_border) &
                (all_anchors[:, 1] >= -_allowed_border) &
                (all_anchors[:, 2] < img_w + _allowed_border) &  # width
                (all_anchors[:, 3] < img_h + _allowed_border)  # height
            )[0]
        else: # so we are doing this (keeping all anchors even if they fall off the image)
            inds_inside = range(all_anchors.shape[0])

        anchors = all_anchors[inds_inside, :]

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = np.empty((len(inds_inside),), dtype=np.float32)
        labels.fill(-1)

        # or here is even better?
        ##########################################################

        # overlaps between the anchors and the gt boxes
        overlaps = bbox_overlaps(
            np.ascontiguousarray(anchors, dtype=np.float),
            np.ascontiguousarray(gt_boxes, dtype=np.float)) #(boxes, query boxes) query_boxes = gt_boxes?

        # here, if overlap is empty, we should do something else
        ########################################################

        argmax_overlaps = overlaps.argmax(axis=1) 
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[
            gt_argmax_overlaps, np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

        if not cfgs.TRAIN_RPN_CLOOBER_POSITIVES: # this is set to false in current cfgs, so we do it here
            labels[max_overlaps < cfgs.RPN_IOU_NEGATIVE_THRESHOLD] = 0

        labels[gt_argmax_overlaps] = 1
        labels[max_overlaps >= cfgs.RPN_IOU_POSITIVE_THRESHOLD] = 1

        if cfgs.TRAIN_RPN_CLOOBER_POSITIVES: # this is set to false in current cfgs
            labels[max_overlaps < cfgs.RPN_IOU_NEGATIVE_THRESHOLD] = 0

        num_fg = int(cfgs.RPN_MINIBATCH_SIZE * cfgs.RPN_POSITIVE_RATE)
        fg_inds = np.where(labels == 1)[0]
        if len(fg_inds) > num_fg:
            disable_inds = npr.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1

        num_bg = cfgs.RPN_MINIBATCH_SIZE - np.sum(labels == 1)
        if is_restrict_bg:
            num_bg = max(num_bg, num_fg * 1.5)
        bg_inds = np.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            disable_inds = npr.choice(
                bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            labels[disable_inds] = -1

        #wth = gt_boxes[argmax_overlaps, :]
        bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])


        # map up to original set of anchors
        labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)

        # labels = labels.reshape((1, height, width, A))
        rpn_labels = labels.reshape((-1, 1))

        # bbox_targets
        bbox_targets = bbox_targets.reshape((-1, 4))
        rpn_bbox_targets = bbox_targets

    else:
        rpn_bbox_targets = all_anchors.reshape((-1, 4))
        
        rpn_labels = np.zeros(len(rpn_bbox_targets),dtype=np.float32)
        num_bg = cfgs.RPN_MINIBATCH_SIZE - np.sum(rpn_labels == 1)
        bg_inds = np.where(rpn_labels == 0)[0]
        if len(bg_inds) > num_bg:
            disable_inds = npr.choice(
                bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            rpn_labels[disable_inds] = -1
        
        #wth = np.zeros(1).astype(np.float32) # just to return something for testing
        #overlaps = np.zeros(1) # just to return something for testing
    
    return rpn_labels, rpn_bbox_targets #, wth, overlaps.astype(np.float32)


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1: # if there is only one dimension. our "lables" variable is only one dimension
        ret = np.empty((count,), dtype=np.float32) # make an empty array of length # of anchors
        ret.fill(fill) # fill it with the value of fill (-1 for this module)
        ret[inds] = data # this will just return all the data in our case
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32) # for our "bbox_targets" case
        ret.fill(fill) # we fill with zeros
        ret[inds, :] = data # and then refill them all with the data, so we aren't changing it it our case?
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""
    # targets = bbox_transform(ex_rois, gt_rois[:, :4]).astype(
    #     np.float32, copy=False)
    targets = encode_and_decode.encode_boxes(unencode_boxes=gt_rois,
                                             reference_boxes=ex_rois,
                                             scale_factors=cfgs.ANCHOR_SCALE_FACTORS)
    # targets = encode_and_decode.encode_boxes(ex_rois=ex_rois,
    #                                          gt_rois=gt_rois,
    #                                          scale_factor=None)
    return targets
