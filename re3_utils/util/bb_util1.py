import numpy as np
import numbers

LIMIT = 99999999


# [x1 y1, x2, y2] to [xMid, yMid, width, height]
def xyxy_to_xywh(bboxes, clipMin=-LIMIT, clipWidth=LIMIT, clipHeight=LIMIT, round=False):
    tbox = np.copy(bboxes)
    tbox = tbox.reshape((1,4))

    obox = numpy.zeros((1,4))
    obox[0] = 1.0 *(tbox[0] + tbox[2] )/2
    obox[1] = 1.0 *(tbox[1] + tbox[3] )/2
    obox[2] = 1.0 *(tbox[2] - tbox[0])
    obox[3] = 1.0 *(tbox[3] - tbox[1])

    for i in range(0,4):
        if obox[i] > LIMIT : 
            obox[i] = LIMIT
        if obox[i] < -LIMIT :
            obox[i] = -LIMIT

    return np.round(obox).astype(int)

# [xMid, yMid, width, height] to [x1 y1, x2, y2]
def xywh_to_xyxy(bboxes, clipMin=-LIMIT, clipWidth=LIMIT, clipHeight=LIMIT, round=False):
    tbox = np.copy(bboxes)
    tbox = tbox.reshape((1,4))

    obox = numpy.zeros((1,4))
    obox[0] = 1.0 *(tbox[0] - tbox[2] )/2
    obox[1] = 1.0 *(tbox[1] - tbox[3] )/2
    obox[2] = 1.0 *(tbox[2] + tbox[0])/2
    obox[3] = 1.0 *(tbox[3] + tbox[1])/2

    for i in range(0,4):
        if obox[i] > LIMIT : 
            obox[i] = LIMIT
        if obox[i] < -LIMIT :
            obox[i] = -LIMIT

    return np.round(obox).astype(int)

# @bboxes {np.array} 4xn array of boxes to be scaled
# @scalars{number or arraylike} scalars for width and height of boxes
# @in_place{bool} If false, creates new bboxes.
def scale_bbox(bboxes, scalars,clipMin=-LIMIT, clipWidth=LIMIT, clipHeight=LIMIT,round=False, in_place=False):
    addedAxis = False
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes, dtype=np.float32)
    if len(bboxes.shape) == 1:
        addedAxis = True
        bboxes = bboxes[:,np.newaxis]
    if isinstance(scalars, numbers.Number):
        scalars = np.full((2, bboxes.shape[1]), scalars, dtype=np.float32)
    if not isinstance(scalars, np.ndarray):
        scalars = np.array(scalars, dtype=np.float32)
    if len(scalars.shape) == 1:
        scalars = np.tile(scalars[:,np.newaxis], (1,bboxes.shape[1])).astype(np.float32)

    bboxes = bboxes.astype(np.float32)

    width = bboxes[2,...] - bboxes[0,...]
    height = bboxes[3,...] - bboxes[1,...]
    xMid = (bboxes[0,...] + bboxes[2,...]) / 2.0
    yMid = (bboxes[1,...] + bboxes[3,...]) / 2.0
    if not in_place:
        bboxesOut = bboxes.copy()
    else:
        bboxesOut = bboxes

    bboxesOut[0,...] = xMid - width * scalars[0,...] / 2.0
    bboxesOut[1,...] = yMid - height * scalars[1,...] / 2.0
    bboxesOut[2,...] = xMid + width * scalars[0,...] / 2.0
    bboxesOut[3,...] = yMid + height * scalars[1,...] / 2.0

    if clipMin != -LIMIT or clipWidth != LIMIT or clipHeight != LIMIT:
        bboxesOut = clip_bbox(bboxesOut, clipMin, clipWidth, clipHeight)
    if addedAxis:
        bboxesOut = bboxesOut[:,0]
    if round:
        bboxesOut = np.round(bboxesOut).astype(np.int32)
    return bboxesOut


def make_square(bboxes, in_place=False):
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes).astype(np.float32)
    if len(bboxes.shape) == 1:
        numBoxes = 1
        width = bboxes[2] - bboxes[0]
        height = bboxes[3] - bboxes[1]
    else:
        numBoxes = bboxes.shape[1]
        width = bboxes[2,...] - bboxes[0,...]
        height = bboxes[3,...] - bboxes[1,...]
    maxSize = np.maximum(width, height)
    scalars = np.zeros((2, numBoxes))
    scalars[0,...] = maxSize * 1.0 / width
    scalars[1,...] = maxSize * 1.0 / height
    return scale_bbox(bboxes, scalars, in_place=in_place)


# Converts from the full image coordinate system to range 0:crop_padding. Useful for getting the coordinates
#   of a bounding box from image coordinates to the location within the cropped image.
# @bbox_to_change xyxy bbox whose coordinates will be converted to the new reference frame
# @crop_location xyxy box of the new origin and max points (without padding)
# @crop_padding the amount to pad the crop_location box (1 would be keep it the same, 2 would be doubled)
# @crop_size the maximum size of the coordinate frame of bbox_to_change.
def to_crop_coordinate_system(bbox_to_change, crop_location, crop_padding, crop_size):
    crop_location = np.array(crop_location,dtype = np.float32)
    bbox_to_change = np.array(bbox_to_change,dtype=np.float32)

    crop_location = scale_bbox(crop_location, crop_padding)
    crop_location_xywh = xyxy_to_xywh(crop_location)
    bbox_to_change -= crop_location[[0,1,0,1]]
    bbox_to_change *= crop_size / crop_location_xywh[[2,3,2,3]]
    return bbox_to_change


# Inverts the transformation from to_crop_coordinate_system
# @crop_size the maximum size of the coordinate frame of bbox_to_change.
def from_crop_coordinate_system(bbox_to_change, crop_location, crop_padding, crop_size):
    crop_location = np.array(crop_location,dtype = np.float32)
    bbox_to_change = np.array(bbox_to_change,dtype=np.float32)

    crop_location = scale_bbox(crop_location, crop_padding)
    crop_location_xywh = xyxy_to_xywh(crop_location)
    bbox_to_change *= crop_location_xywh[[2,3,2,3]] / crop_size
    bbox_to_change += crop_location[[0,1,0,1]]
    return bbox_to_change


