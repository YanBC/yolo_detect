# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np


class YoloV3(object):
    '''
    Yolov3 classification wrapper with opencv backend (cv.dnn.readNetFromDarknet)

    ATTRIBUTE:
        net <cv.dnn_Net>: opencv darnet model

        inpWidth <int>: yolov3 input image width, should be
                            consistant with darknet configuration 
                            file

        inpHeight <int>: yolov3 input image height, should be
                            consistant with darknet configuration 
                            file

        confThreshold <float>: classes classification confidence 
                                threshold

        nmsThreshold <float>: nms threshold

    FUNCTION:
        inference()

        predict()

    '''
    def __init__(self, config, weights, inpWidth, inpHeight, confThreshold, nmsThreshold):
        '''
        Initialization

        INPUT:
            config <string>: path to darknet configuration file

            weights <string>: path to darknet model weights

            inpWidth <int>: yolov3 input image width, should be
                            consistant with darknet configuration 
                            file

            inpHeight <int>: yolov3 input image height, should be
                            consistant with darknet configuration 
                            file

            confThreshold <float>: classes classification confidence 
                                    threshold

            nmsThreshold <float>: nms threshold
        '''
        # create network 
        self.net = cv.dnn.readNetFromDarknet(config, weights)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)                                    # Try to configure GPU later
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

        self.inpWidth = inpWidth
        self.inpHeight = inpHeight
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold


    def _postprocess(self, frame, outs, confThreshold, nmsThreshold):
        '''
        Perform post-prediction processing, produce the final 
        bounding box predictions

        INPUT:
            frame <numpy array>: target image in shape (height, width, channel),
                                    where height, width and channel should be 
                                    consistant with darknet configuration

            outs <list>: outputs from net.forward(); yolov3 predicts at 3 
                        different scales, so outs should be a list of 3
                        2D arrays; in the orginal yolov3, these 2D arrays
                        should be of the following shapes:
                        1. (507, (4 + 1 + num_of_classes)), where 
                            507 = 13 * 13 * 3
                        2. (2028, (4 + 1 + num_of_classes)), where
                            2028 = 26 * 26 * 3
                        3. (32448, (4 + 1 + num_of_classes)), where
                            32448 = 104 * 104 * 3

            confThreshold <float>: prediction confidence threshold

            nmsThreshold <float>: non-max-suppression threshold

        OUTPUT:
            ret_c <list>: prediction confidences

            ret_cls <list>: prediction classes

            ret_b <list>: prediction bounding boxes, each in the
                            order of (x1, y1, width, height), where
                            (x1 ,y1) is the top-left coordinates

        '''

        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        '''
        Scan through all the bounding boxes output 
        from the network and keep only the ones 
        with high confidence scores. Assign the box's 
        class label as the class with the highest score.
        '''
        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId] # * detection[4]                                      # not sure if I am doing the right thing here
                if confidence > confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        '''
        Perform non maximum suppression to eliminate 
        redundant overlapping boxes with lower confidences.
        '''
        indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)                 # BreakPoint: check indices type
        
        ret_c = []
        ret_cls = []
        ret_b = []

        if len(indices) > 0:
            tmp = indices[:,0]
            ret_c = [confidences[i] for i in tmp]
            ret_cls = [classIds[i] for i in tmp]
            ret_b = [boxes[i] for i in tmp]

        return ret_c, ret_cls, ret_b


    def _inference(self, frame):
        '''
        Inference on a given image

        INPUT:
            frame <numpy array>: target image in shape (height, width, channel),
                                    where height, width and channel should be 
                                    consistant with darknet configuration

        OUTPUT:
            outs <list>: outputs from net.forward(); yolov3 predicts at 3 
                        different scales, so outs should be a list of 3
                        2D arrays; in the orginal yolov3, these 2D arrays
                        should be of the following shapes:
                            1. (507, (4 + 1 + num_of_classes)), where 
                                507 = 13 * 13 * 3
                            2. (2028, (4 + 1 + num_of_classes)), where
                                2028 = 26 * 26 * 3
                            3. (32448, (4 + 1 + num_of_classes)), where
                                32448 = 104 * 104 * 3                 
        '''

        # get 4-D blob from frame
        blob = cv.dnn.blobFromImage(frame, 1/255, (self.inpWidth, self.inpHeight), [0,0,0], 1, crop=False)
        self.net.setInput(blob)

        # inference
        layersNames = self.net.getLayerNames()
        outNames = [layersNames[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        outs = self.net.forward(outNames)

        return outs     


    def predict(self, frame):
        '''
        Make a prediction on the given image

        INPUT:
            frame <numpy array>: target image in shape (height, width, channel),
                                    where height, width and channel should be 
                                    consistant with darknet configuration

        OUTPUT:
            ret <list>: predictions, each elements is a bounding box prediction
                            in the following format:
                                (confidence, class, [x1, y1, width, height])
                            where (x1, y1) is the top-left coordinates
        '''

        # inference
        outs = self._inference(frame)

        # post processing
        confidences, classes, boxes = self._postprocess(frame, outs, self.confThreshold, self.nmsThreshold)

        ret = []
        for i in range(len(boxes)):
            # tmp = [confidences[i]] + [classes[i]] + boxes[i]
            tmp = (confidences[i], classes[i], boxes[i])
            ret.append(tmp)

        return ret







