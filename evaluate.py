# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2 as cv
from nn import YoloV3


imageExts = ['.jpg']
annoExts = ['.txt']


def _IoU(boxA, boxB):
    '''
    Calculate the intersection over union ratio between two boxes
    '''

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def getPredictions(model, dirPath):
    '''
    make predictions on all images in <dirPath>

    INPUT:
        model <YoloV3>: the to-be-evaluated model 

        dirPath <string>: path to image and annotation directory


    OUTPUT:
        ret <dictionary>: Key <string>: 
                            image file name
                          
                          Value <list>: 
                            annotations in the form of [anno1, anno2, ..., annoN]
                                
                            anno <tuple>:
                                (confidence <float>, class <int>, 
                                    [left <int>, top <int>, right <int>, bottom <int>])


    NOTES:
        A typical yolo annotation direcotry should be like:
        |
        ├── frame126.jpg
        ├── frame126.txt
        ├── frame127.jpg
        ├── frame127.txt
        ├── frame128.jpg
        ├── frame128.txt
        ├── frame129.jpg
        ├── frame129.txt
        ├── frame130.jpg
        ├── frame130.txt
        ├── frame131.jpg
        ├── frame131.txt
        ├── frame132.jpg
        ├── frame132.txt
        ├── frame133.jpg    
    '''
    files = os.listdir(dirPath)
    images = [f for f in files if (imageExts[0] in f)]

    ret = dict()
    for image in images:
        # read image
        imagePath = os.path.join(dirPath, image)
        img = cv.imread(imagePath)

        tmp = []
        boxes = model.predict(img)
        for box in boxes:
            confidence = float(box[0])
            classId = int(box[1])
            left = int(box[2][0])
            top = int(box[2][1])
            right = int(box[2][0] + box[2][2])
            bottom = int(box[2][1] + box[2][3])

            tmp.append((confidence, classId, [left, top, right, bottom]))
        ret[image] = tmp

    return ret


def getAnnotations(dirPath):
    '''
    Read yolo annotations

    INPUT:
        dirPath <string>: path to image and annotation directory


    OUTPUT:
        ret <dictionary>: Key <string>: 
                            image file name
                          
                          Value <list>: 
                            annotations in the form of [anno1, anno2, ..., annoN]
                                
                            anno <tuple>:
                                (class <int>, [left <int>, top <int>, right <int>, bottom <int>])


    NOTES:
        A typical yolo annotation direcotry should be like:
        |
        ├── frame126.jpg
        ├── frame126.txt
        ├── frame127.jpg
        ├── frame127.txt
        ├── frame128.jpg
        ├── frame128.txt
        ├── frame129.jpg
        ├── frame129.txt
        ├── frame130.jpg
        ├── frame130.txt
        ├── frame131.jpg
        ├── frame131.txt
        ├── frame132.jpg
        ├── frame132.txt
        ├── frame133.jpg
    '''
    files = os.listdir(dirPath)
    images = [f for f in files if (imageExts[0] in f)]
    annoFiles = [f for f in files if (annoExts[0] in f)]

    ret = dict()
    for annoFile in annoFiles:

        # get image informations
        imageName = annoFile.split('.')[0] + '.jpg'
        imagePath = os.path.join(dirPath, imageName)
        image = cv.imread(imagePath)
        if image is None:
            print('Error in getAnnotations(%s): %s not exist' % (dirPath, imagePath))
            continue
        height, width, _ = image.shape

        # read annotation
        ret[imageName] = []
        with open(os.path.join(dirPath, annoFile), 'r') as tmp:
            line = tmp.readline().strip()
            while (line != ''):
                c, x, y, w, h = line.split(' ')
                tmp_c = int(c)
                tmp_x = float(x) * width
                tmp_y = float(y) * height
                tmp_w = float(w) * width
                tmp_h = float(h) * height

                left = int(tmp_x - tmp_w / 2)
                top = int(tmp_y - tmp_h / 2)
                right = int(tmp_x + tmp_w / 2)
                bottom = int(tmp_y + tmp_h / 2)
                
                ret[imageName].append((tmp_c, [left, top, right, bottom]))

                line = tmp.readline().strip()

    return ret


def _findMaxIoU(target, gt_list):
    '''
    Find the box that has the maximun IoU with target

    INPUT:
        target <list>: coordinates of the target box in the form of
                            [left, top, right, bottom]

        gt_list <list>: a list of boxes that have the same form of
                            target

    OUTPUT:
        max_iou <float>: the found maximum IoU between target box 
                            and boxes in gt_list

        gt_index <int>: the index of the box in gt_list that has
                            the maximum IoU with target
    '''
    max_iou = 0
    gt_index = -1
    for i in range(len(gt_list)):
        tmp_gt = gt_list[i]
        tmp_iou = _IoU(tmp_gt, predition)
        if tmp_iou > max_iou:
            max_iou = tmp_iou
            gt_index = i

    return gt_index, max_iou        # return -1,0 if not found


def _maxIoUSuppression(iou_mat, iouThres=0.5):
    '''
    Assign each predicted box to ground true boxes based on IoU

    INPUT:
        iou_mat <2d array>: IoU matrix. Entry_i,j is the IoU ratio 
                            of the i_th ground true box and the 
                            j_th predicted box

        iouThres <float>: IoU threshold. Any predicted box that has
                            an IoU lower that this threshold will 
                            be considered false positive

    OUTPUT:
        gt_indices <1d array>: a 1d array of length iou_mat.shape[1],
                                gt_indices[j] is the index of ground
                                true box assigned to predicted box j;
                                gt_indices[j] = -1 if no ground true
                                box is assigned
    '''
    gt_indices = np.argmax(iou_mat, axis=0)

    for j in range(len(gt_indices)):
        gt_index = gt_indices[j]

        # if iou(gt, pd) is less than iouThres,
        # the corresponding pd is a false positive
        if iou_mat[gt_index, j] < iouThres:
            gt_indices[j] = -1
            continue

        # check if current prediction is matched with a 
        # ground-true box that has already been assigned
        if gt_index in set(gt_indices[:j]):
            prev_j = np.where(gt_indices[:j]==gt_index)[0][0]
            if iou_mat[gt_index, prev_j] > iou_mat[gt_index, j]:
                gt_indices[j] = -1
            else:
                gt_indices[prev_j] = -1

    return gt_indices


def getStats(classId, y_pred, y_true, iouThres=0.5):
    '''
    Perform statistical analysis on bounding box predictions

    INPUT:
        classId <int>: the class to be analyzed

        y_pred <dictionary>: predictions; output from getPredictions()

        y_true <dictionary>: ground-true; output from getAnnotations()

        iouThres <float>: IoU threshold for positive predictions

    OUTPUT:
        tp_size <int>: number of true positives

        fp_size <int>: number of false positves

        fn_size <int>: number of false negatives

        ret_tp <list>: true positives information. each element is a true 
                        positive's confidence and its IoU with groud true:
                            [confidence, IoU]
    '''
    fn_size = 0
    fp_size = 0
    tp_size = 0
    ret_tp = []

    # make sure images in y_pred are the same with images in y_true
    assert y_pred.keys() == y_true.keys()
    images = sorted(y_pred.keys())

    # loop over each image
    for image in images:
        gt_list = [box[1] for box in y_true[image] if box[0] == classId]
        pd_list = [box[2] for box in y_pred[image] if box[1] == classId]

        # if there is no predicted boxes, all ground true
        # boxes would go to the false negative category
        if len(pd_list) == 0:
            fn_size += len(gt_list)
            continue

        # if there is no ground true boxes, all predicted 
        # boxes would be false negatives
        if len(gt_list) == 0:
            fp_size += len(pd_list)
            continue

        # get iou matrix
        iou = np.empty((len(gt_list), len(pd_list)))
        for i in range(len(gt_list)):
            for j in range(len(pd_list)):
                iou[i,j] = _IoU(gt_list[i], pd_list[j])

        # decide whether a predicted box is true positive 
        # or false positive
        gt_assigned = _maxIoUSuppression(iou, iouThres)

        # record all true positives to ret_tp
        preds = np.where(gt_assigned != -1)[0]
        for pred in preds:
            tmp_confidence = y_pred[image][pred][0]
            tmp_iou = iou[gt_assigned[pred], pred]

            ret_tp.append([tmp_confidence, tmp_iou])

        # record false positives and false negatives
        fp_size += len(pd_list) - len(preds)
        fn_size += len(gt_list) - len(preds)

    # yes, I am that obsessive-compulsive
    tp_size = len(ret_tp)
    fp_size = int(fp_size)
    fn_size = int(fn_size)
    return tp_size, fp_size, fn_size, ret_tp




if __name__ == '__main__':
    import argparse
    import yaml
    import pickle
    import shutil
    import sys

    parser = argparse.ArgumentParser(description='YOLO Detection Evaluation')
    parser.add_argument('dirPath', help='Path to yolo annotation directory')
    parser.add_argument('--configs', help='Path to configuration file (yaml). Default: "./detector.yml"')
    parser.add_argument('--iouThres', type=float, default=0.5, help='IoU Threshold. Default: 0.5')
    args = parser.parse_args()

    imageDir = args.dirPath

    # load configurations
    if args.configs:
        with open(args.configs, 'r') as f:
            configs = yaml.safe_load(f)
    else:
        configPath = './detector.yml'
        with open(configPath, 'r') as f:
            configs = yaml.safe_load(f)
    confThreshold = configs['confThreshold']
    nmsThreshold = configs['nmsThreshold']
    inpWidth = configs['inpWidth']
    inpHeight = configs['inpHeight']
    classesFile = configs['classesFile']
    modelConfiguration = configs['modelConfiguration']
    modelWeights = configs['modelWeights']


    # try to load annos and preds from cach
    if sys.argv[0].rfind('/') == -1:
        codeLocale = './'
    else:
        codeLocale = sys.argv[0][:sys.argv[0].rfind('/')]
    cachDir = os.path.join(codeLocale, 'cach_eval')
    cachFile = 'cach.pkl'
    old_imageDir = None

    if os.path.isfile(os.path.join(cachDir, 'cach.pkl')):
        with open(os.path.join(cachDir, 'cach.pkl'), 'br') as f:
            old_imageDir = pickle.load(f)
            old_annos = pickle.load(f)
            old_preds = pickle.load(f)
    if old_imageDir == imageDir:
        annos = old_annos
        preds = old_preds
    else:
        # get annotations
        print('# READING ANNOTATIONS ...')
        annos = getAnnotations(imageDir)
        print('# DONE!\n')

        # get predictions
        print('# MAKING PREDICTIONS ...')
        net = YoloV3(modelConfiguration, modelWeights, inpWidth, inpHeight, confThreshold, nmsThreshold)
        preds = getPredictions(net, imageDir)
        print('# DONE!\n')

        # write to cach
        if os.path.isdir(cachDir):
            shutil.rmtree(cachDir)
        os.mkdir(cachDir)
        with open(os.path.join(cachDir, cachFile), 'bw') as f:
            pickle.dump(imageDir, f)
            pickle.dump(annos, f)
            pickle.dump(preds, f)


    # performance analysis
    print('# ANALYSIS RESULTS')
    print('Using IoU threshold of %0.2f\n' % (args.iouThres))
    classNames = []
    with open(classesFile, 'r') as f:
        line = f.readline().strip()
        while line:
            classNames.append(line)
            line = f.readline().strip()

    for c in range(len(classNames)):
        tps, fps, fns, tp_info = getStats(c, preds, annos, args.iouThres)
        tmp = np.array(tp_info)
        try:
            assert tmp.shape[1] == 2
        except IndexError:
            continue

        tmp_avgs = tmp.mean(axis=0)

        print('Class: %s' % classNames[c])
        print('Recall: %0.3f' % (tps/(tps+fns)))
        print('Precision: %0.3f' % (tps/(tps+fps)))
        print('True Positive:')
        print('    Average Confidence: %0.3f' % (tmp_avgs[0]))
        print('    Average IoU: %0.3f\n' % (tmp_avgs[1]))
