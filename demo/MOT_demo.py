import cv2
import glob
import numpy as np
import sys
import os.path
import matplotlib.pyplot as plt
from skimage import io
from sklearn.utils.linear_assignment_ import linear_assignment
basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))
from tracker import re3_tracker
from tracker import network
import matplotlib.patches as patches  


coun = 1
trackers = []
colours = np.random.rand(32,3) # used only for display
trackers = []
firstli = []
dictt = {}
firstli = []
keepingalldets = []
bboxes = []
bboxes2 = []

def iou(bb_test, bb_gt):
  """
  Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  """
  xx1 = np.maximum(bb_test[0], bb_gt[0])
  yy1 = np.maximum(bb_test[1], bb_gt[1])
  xx2 = np.minimum(bb_test[2], bb_gt[2])
  yy2 = np.minimum(bb_test[3], bb_gt[3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
    + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
  return (o)

def associate_detections_to_trackers(detections, trackers, iou_threshold = 0.00000001):
  """
  Assigns detections to tracked object (both represented as bounding boxes)
  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if( len(trackers)==0 ):
    return np.empty((0,2), dtype=int), np.arange(len(detections)), np.empty((0,5), dtype=int)
  
  iou_matrix = np.zeros((len(detections),len(trackers)), dtype=np.float32)
  for d,det in enumerate(detections):
    for t,trk in enumerate(trackers):
      iou_matrix[d,t] = iou(det, trk)
  matched_indices = linear_assignment(-iou_matrix)

  unmatched_detections = []
  for d,det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t,trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if( iou_matrix[m[0],m[1]]<iou_threshold ):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))

  print matches

  if(len(matches)==0):
    matches = np.empty((0,2), dtype=int)
  else:
    matches = np.concatenate(matches, axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
 
#Main function begin here 
tf_vars = {}
tf.Graph().as_default()
tf_vars['imagePlaceholder'] = tf.placeholder(tf.uint8, shape=(None, 227, 227, 3))
tf_vars['prevLstmState'] = tuple([tf.placeholder(tf.float32, shape=(None, 1024)) for _ in range(4)])
tf_vars['batch_size'] = tf.placeholder(tf.int32, shape=())
tf_vars['outputs'],tf_vars['state1'],tf_vars['state2'] = network.inference(tf_vars['imagePlaceholder'], num_unrolls=1, batch_size=tf_vars['batch_size'], train=False,prevLstmState=tf_vars['prevLstmState'])
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf_vars['sess'] = tf.Session(config=config)
ckpt = tf.train.get_checkpoint_state(os.path.join(basedir, '..', '../logs/', 'checkpoints'))
tf_util.restore(sess, ckpt.model_checkpoint_path)
tf_vars['tracked_data'] = {}
tf_vars['total_forward_count'] = -1


image_paths = sorted(glob.glob(os.path.join(os.path.dirname(__file__), 'data', '*.jpg')))
seq = '/home/deepak/Desktop/Ass2/re3-tensorflow/MOT17-01-DPM'
seq_dets = np.loadtxt('%s/det/det.txt'%(seq), delimiter=',')

plt.ion()
fig = plt.figure()

for frame in range(int(seq_dets[:,0].max())):
  frame += 1 # detection and frame numbers begin at 1
  dets = seq_dets[seq_dets[:,0]==frame,2:6]
  dets[:, 2:4] += dets[:, 0:2] # convert to [x1,y1,w,h] to [x1,y1,x2,y2]

  ax1 = fig.add_subplot(111, aspect='equal')
  fn = seq + '/img1/%06d.jpg'%(frame)
  im = io.imread(fn)
  ax1.imshow(im)
  imageRGB = im
  unmatched_trks = []
  
  if( len(firstli) ):
    bboxes2 = []

    #print '0'
    bboxes = tracker.multi_track(tf_vars,firstli, imageRGB)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, bboxes)
    #print '1'
        
    # Create and initialise new trackers for unmatched detections
    firstli2 = []
    dictt2 = {}
    for udets in unmatched_dets:
        firstli2.append(coun)
        firstli.append(coun)
        obj = dets[udets,:]
        secondli = []
        for tocomma in obj:
           secondli.append(tocomma)
        dictt2[coun] = secondli
        dictt[coun] = secondli
        coun += 1
    
    bboxes = tracker.multi_track(tf_vars,firstli, imageRGB, dictt2) 
    #print '2'

    # Update matched trackers with assigned detections
    for t, trk in enumerate(bboxes):
      if(t not in unmatched_trks):
        d = matched[np.where(matched[:,1]==t)[0],0]
        trk = dets[d,:]
    #print '3'

    #print 'Umatched ', len(firstli), len(bboxes), len(bboxes2), len(unmatched_trks)
    unmatched_trks.sort() 
    for udets in reversed(unmatched_trks):
        #print udets, len(unmatched_trks)
        ind = firstli[udets]
        del firstli[udets]
        del dictt[ind]
        bboxes = np.delete(bboxes, udets, axis=0)
    #print 'Umatched2 ', len(firstli), len(bboxes), len(bboxes2)
    #print '4'   

  else:

    for obj in dets:
       secondli = []
       firstli.append(coun)
       for tocomma in obj:
          secondli.append(tocomma)
       dictt[coun] = secondli 
       coun += 1
    bboxes = tracker.multi_track(tf_vars,firstli, imageRGB, dictt)

  ele = 0
  #print len(bboxes), len(bboxes2), len(firstli)
  for bb, bbox in enumerate(bboxes):
      listele = []
      listele.append(int(bbox[0]))
      listele.append(int(bbox[1]))
      listele.append(int(bbox[2]))
      listele.append(int(bbox[3]))
      dictt[ firstli[ele] ] = listele
      bbox = bbox.astype(np.int32)
      ax1.add_patch(patches.Rectangle((int(bbox[0]), int(bbox[1])), int(bbox[2])-int(bbox[0]), int(bbox[3])-int(bbox[1]), fill=False, lw=3, ec=colours[firstli[ele]%32,:]))
      ax1.set_adjustable('box-forced')
      ele += 1  
   
  fig.canvas.flush_events()
  plt.draw()
  ax1.cla()
