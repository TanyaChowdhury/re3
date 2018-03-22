import cv2
import glob
import numpy as np
import os
import tensorflow as tf

import sys

CROP_SIZE = 227
CROP_PAD = 2
MAX_TRACK_LENGTH = 32
LSTM_SIZE = 1024

SPEED_OUTPUT = True

from re3_utils.util import bb_util1
from re3_utils.util import im_util1
from re3_utils.tensorflow_util import tf_util

def track(tf_vars, unique_id, image, starting_box=None):

    img = img.copy()

    if starting_box is not None:
        lstmState = [np.zeros((1, LSTM_SIZE)) for _ in range(4)]
        pastBBox = np.array(starting_box) # turns list into numpy array if not and copies for safety.
        prevImage = img
        originalFeatures = None
        forwardCount = 0
    elif unique_id in tf_vars['tracked_data']:
        lstmState, pastBBox, prevImage, originalFeatures, forwardCount = tf_vars['tracked_data'][unique_id]
    else:
        raise Exception('Unique_id %s with no initial bounding box' % unique_id)

    croppedInput0, pastBBoxPadded = im_util.get_cropped_input(prevImage, pastBBox, CROP_PAD, CROP_SIZE)
    croppedInput1,_ = im_util.get_cropped_input(img, pastBBox, CROP_PAD, CROP_SIZE)
    feed_dict = {
            tf_vars['imagePlaceholder'] : [croppedInput0, croppedInput1],
            tf_vars['prevLstmState'] : lstmState,
            tf_vars['batch_size'] : 1,
            }
    rawOutput, s1, s2 = tf_vars['sess'].run([tf_vars['outputs'], tf_vars['state1'], tf_vars['state2']], feed_dict=feed_dict)
    lstmState = [s1[0], s1[1], s2[0], s2[1]]
    if forwardCount == 0:
        originalFeatures = [s1[0], s1[1], s2[0], s2[1]]

    prevImage = img

    # Shift output box to full image coordinate system.
    outputBox = bb_util.from_crop_coordinate_system(rawOutput.squeeze() / 10.0, pastBBoxPadded, 1, 1)
    if forwardCount > 0 and forwardCount % MAX_TRACK_LENGTH == 0:
        croppedInput, _ = im_util.get_cropped_input(image, outputBox, CROP_PAD, CROP_SIZE)
        input = np.tile(croppedInput[np.newaxis,...], (2,1,1,1))
        feed_dict = {
                tf_vars['imagePlaceholder'] : input,
                tf_vars['prevLstmState'] : originalFeatures,
                tf_vars['batch_size'] : 1,
                }
        rawOutput, s1, s2 = tf_vars['sess'].run([tf_vars['outputs'], tf_vars['state1'], tf_vars['state2']], feed_dict=feed_dict)
        lstmState = [s1[0], s1[1], s2[0], s2[1]]

    forwardCount += 1
    tf_vars['total_forward_count'] += 1

    if starting_box is not None:
            # Use label if it's given
        outputBox = np.array(starting_box)

    tf_vars['tracked_data'][unique_id] = (lstmState, outputBox, image, originalFeatures, forwardCount)


    return outputBox


def multi_track(tf_vars, unique_ids, image, starting_boxes=None):

    img = img.copy()

    # Get inputs for each track.
    images = []
    lstmStates = [[] for _ in range(4)]
    pastBBoxesPadded = []
    if starting_boxes is None:
        starting_boxes = dict()
    for unique_id in unique_ids:
        if unique_id in starting_boxes:
            lstmState = [np.zeros((1, LSTM_SIZE)) for _ in range(4)]
            pastBBox = np.array(starting_boxes[unique_id]) # turns list into numpy array if not and copies for safety.
            prevImage = image
            originalFeatures = None
            forwardCount = 0
            tf_vars['tracked_data'][unique_id] = (lstmState, pastBBox, image, originalFeatures, forwardCount)
        elif unique_id in tf_vars['tracked_data']:
            lstmState, pastBBox, prevImage, originalFeatures, forwardCount = tf_vars['tracked_data'][unique_id]
        else:
            raise Exception('Unique_id %s with no initial bounding box' % unique_id)

        croppedInput0, pastBBoxPadded = im_util.get_cropped_input(prevImage, pastBBox, CROP_PAD, CROP_SIZE)
        croppedInput1,_ = im_util.get_cropped_input(image, pastBBox, CROP_PAD, CROP_SIZE)
        pastBBoxesPadded.append(pastBBoxPadded)
        images.extend([croppedInput0, croppedInput1])
        for ss,state in enumerate(lstmState):
            lstmStates[ss].append(state.squeeze())

    lstmStateArrays = []
    for state in lstmStates:
        lstmStateArrays.append(np.array(state))

    feed_dict = { tf_vars['imagePlaceholder'] : images, tf_vars['prevLstmState'] : lstmStateArrays, tf_vars['batch_size'] : len(images)/2,}
    rawOutput, s1, s2 = tf_vars['sess'].run([tf_vars['outputs'], tf_vars['state1'], tf_vars['state2']], feed_dict=feed_dict)
    outputBoxes = np.zeros((len(unique_ids), 4))
    for uu,unique_id in enumerate(unique_ids):
        lstmState, pastBBox, prevImage, originalFeatures, forwardCount = tf_vars['tracked_data'][unique_id]
        lstmState = [s1[0][[uu],:], s1[1][[uu],:], s2[0][[uu],:], s2[1][[uu],:]]
        if forwardCount == 0:
            originalFeatures = [s1[0][[uu],:], s1[1][[uu],:], s2[0][[uu],:], s2[1][[uu],:]]

        prevImage = image

        # Shift output box to full image coordinate system.
        pastBBoxPadded = pastBBoxesPadded[uu]
        outputBox = bb_util.from_crop_coordinate_system(rawOutput[uu,:].squeeze() / 10.0, pastBBoxPadded, 1, 1)

        if forwardCount > 0 and forwardCount % MAX_TRACK_LENGTH == 0:
            croppedInput, _ = im_util.get_cropped_input(image, outputBox, CROP_PAD, CROP_SIZE)
            input = np.tile(croppedInput[np.newaxis,...], (2,1,1,1))
            feed_dict = { tf_vars['imagePlaceholder'] : input, tf_vars['prevLstmState'] : originalFeatures, tf_vars['batch_size'] : 1,}

            _, s1_new, s2_new = tf_vars['sess'].run([tf_vars['outputs'], tf_vars['state1'], tf_vars['state2']], feed_dict=feed_dict)
            lstmState = [s1_new[0], s1_new[1], s2_new[0], s2_new[1]]

        forwardCount += 1
        tf_vars['total_forward_count'] += 1

        if unique_id in starting_boxes:
            # Use label if it's given
            outputBox = np.array(starting_boxes[unique_id])

        outputBoxes[uu,:] = outputBox
        tf_vars['tracked_data'][unique_id] = (lstmState, outputBox, image, originalFeatures, forwardCount)


    return outputBoxes








