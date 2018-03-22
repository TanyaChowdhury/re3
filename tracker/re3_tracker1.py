#imageName : (String) Name of Image to be read
#startCoords : [ {X1 , Y1 , X2  , Y2} , {X1 , Y1, X2, Y2 }] (Only if initial frame) (list of tuples)
#nObjs : Number of objects to be tracked
def track(imageName,nObjs,startCoords = []):

    basedir = os.path.dirname(__file__)
    tf.Graph().as_default()

    img = cv2.imread()
    images = []
    lstmStates = [[],[],[],[]]
    pastBBoxesPadded = []

    for i in range(0,nObjs):
        if(len(startCoords) > i):
            lstmState = [np.zeros((1, LSTM_SIZE)) for _ in range(4)]
            pastBBox = startCoords[i]
            tracked_data.append(lstmState,pastBBox,img.copy(),None,0)
        elif len(tracked_data) > i:
            lstmState , pastBBox , prevImage , originalFeatures, forwardCount = tracked_data[i]
        else:
            print('No bounding box found for object',i)
            exit(0)

        croppedInput0, pastBBoxPadded = im_util.get_cropped_input(prevImage, pastBBox, CROP_PAD, CROP_SIZE)
        croppedInput1,_ = im_util.get_cropped_input(image, pastBBox, CROP_PAD, CROP_SIZE)
        pastBBoxesPadded.append(pastBBoxPadded)
        images.extend([croppedInput0, croppedInput1])
        for ss,state in enumerate(lstmState):
            lstmStates[ss].append(state.squeeze())

    lstmStateArrays = []
    for state in lstmStates:
        lstmStateArrays.append(np.array(state))

    feed_dict = {
            self.imagePlaceholder : images,
            self.prevLstmState : lstmStateArrays,
            self.batch_size : len(images) / 2
    }
    
    rawOutput, s1, s2 = self.sess.run([self.outputs, self.state1, self.state2], feed_dict=feed_dict)
    outputBoxes = np.zeros((len(unique_ids), 4))
    for uu,unique_id in enumerate(unique_ids):
        lstmState, pastBBox, prevImage, originalFeatures, forwardCount = self.tracked_data[unique_id]
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
            feed_dict = {
                    self.imagePlaceholder : input,
                    self.prevLstmState : originalFeatures,
                    self.batch_size : 1,
                }
            _, s1_new, s2_new = self.sess.run([self.outputs, self.state1, self.state2], feed_dict=feed_dict)
            lstmState = [s1_new[0], s1_new[1], s2_new[0], s2_new[1]]

        forwardCount += 1
        self.total_forward_count += 1

        if unique_id in starting_boxes:
            # Use label if it's given
            outputBox = np.array(starting_boxes[unique_id])

        outputBoxes[uu,:] = outputBox
        self.tracked_data[unique_id] = (lstmState, outputBox, image, originalFeatures, forwardCount)
   
        return outputBoxes




