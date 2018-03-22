import cv2
import numpy as np

# @inputImage{ndarray HxWx3} Full input image.
# @bbox{ndarray or list 4x1} bbox to be cropped in x1,y1,x2,y2 format.
# @padScale{number} scalar representing amount of padding around image.
#   padScale=1 will be exactly the bbox, padScale=2 will be 2x the input image.
# @outputSize{number} Size in pixels of output crop. Crop will be square and
#   warped.
# @return{tuple(patch, outputBox)} the output patch and bounding box
#   representing its coordinates.
def get_cropped_input(inputImage, bbox, padScale, outputSize):
    bbox = np.array(bbox)
    width = float(bbox[2] - bbox[0])
    height = float(bbox[3] - bbox[1])
    imShape = np.array(inputImage.shape)
    if len(imShape) < 3:
        inputImage = inputImage[:,:,np.newaxis]
    xC = float(bbox[0] + bbox[2]) / 2
    yC = float(bbox[1] + bbox[3]) / 2
    boxOn = np.zeros(4)
    boxOn[0] = float(xC - padScale * width / 2)
    boxOn[1] = float(yC - padScale * height / 2)
    boxOn[2] = float(xC + padScale * width / 2)
    boxOn[3] = float(yC + padScale * height / 2)
    outputBox = boxOn.copy()
    boxOn = np.round(boxOn).astype(int)
    boxOnWH = np.array([boxOn[2] - boxOn[0], boxOn[3] - boxOn[1]])
    imagePatch = inputImage[max(boxOn[1], 0):min(boxOn[3], imShape[0]),
            max(boxOn[0], 0):min(boxOn[2], imShape[1]), :]
    boundedBox = np.clip(boxOn, 0, imShape[[1,0,1,0]])
    boundedBoxWH = np.array([boundedBox[2] - boundedBox[0], boundedBox[3] - boundedBox[1]])

    if imagePatch.shape[0] == 0 or imagePatch.shape[1] == 0:
        patch = np.zeros((int(outputSize), int(outputSize), 3))
    else:
        patch = cv2.resize(imagePatch, (
            max(1, int(np.round(outputSize * boundedBoxWH[0] / boxOnWH[0]))),
            max(1, int(np.round(outputSize * boundedBoxWH[1] / boxOnWH[1])))))
        if len(patch.shape) < 3:
            patch = patch[:,:,np.newaxis]
        patchShape = np.array(patch.shape)

        pad = np.zeros(4, dtype=int)
        pad[:2] = np.maximum(0, -boxOn[:2] * outputSize / boxOnWH)
        pad[2:] = outputSize - (pad[:2] + patchShape[[1,0]])

        if np.any(pad != 0):
            if len(pad[pad < 0]) > 0:
                patch = np.zeros((int(outputSize), int(outputSize), 3))
            else:
                patch = np.lib.pad(
                        patch,
                        ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)),
                        'constant', constant_values=0)
    return patch, outputBox

def get_image_size(fname):
    import struct, imghdr, re

    while True:
        with open(fname, 'rb') as fhandle:
            head = fhandle.read(32)
            if len(head) != 32:
                break
            if imghdr.what(fname) == 'png':
                check = struct.unpack('>i', head[4:8])[0]
                if check != 0x0d0a1a0a:
                    break
                width, height = struct.unpack('>ii', head[16:24])
            elif imghdr.what(fname) == 'gif':
                width, height = struct.unpack('<HH', head[6:10])
            return width, height
    imShape = cv2.imread(fname).shape
    return imShape[1], imShape[0]
