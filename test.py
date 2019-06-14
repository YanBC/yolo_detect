# -*- coding: utf-8 -*-

from nn import YoloV3
import argparse
import yaml
import os
import cv2 as cv
import numpy as np


# Draw the predicted bounding box
def drawPred(conf, classId, left, top, right, bottom, classes=None):
    # Draw a bounding box.
    #    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)

    label = '%.2f' % conf
        
    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (0, 0, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Object Detection with YOLO in OPENCV')
    parser.add_argument('configs', help='Path to configuration file (yaml)')
    parser.add_argument('--image', help='Path to image file.')
    parser.add_argument('--video', help='Path to video file.')
    args = parser.parse_args()


    # load configurations
    # see configuration files for explaination of each parameters
    with open(args.configs, 'r') as f:
        configs = yaml.safe_load(f)
    confThreshold = configs['confThreshold']
    nmsThreshold = configs['nmsThreshold']
    inpWidth = configs['inpWidth']
    inpHeight = configs['inpHeight']
    classesFile = configs['classesFile']
    modelConfiguration = configs['modelConfiguration']
    modelWeights = configs['modelWeights']


    # Process input stream: image or video
    outputFile = "yolo_out_py.avi"
    if (args.image):
        # Open the image file
        if not os.path.isfile(args.image):
            print("Input image file ", args.image, " doesn't exist")
            sys.exit(1)
        cap = cv.VideoCapture(args.image)
        outputFile = args.image[:-4]+'_yolo_out_py.jpg'
    elif (args.video):
        # Open the video file
        if not os.path.isfile(args.video):
            print("Input video file ", args.video, " doesn't exist")
            sys.exit(1)
        cap = cv.VideoCapture(args.video)
        outputFile = args.video[:-4]+'_yolo_out_py.avi'
    else:
        # # Webcam input
        # cap = cv.VideoCapture(0)
        print('You have to pass either [--image] or [--video]')
        print('See <python3 %s -h> for usage' % (sys.argv[0]))
        sys.exit(1)


    # Create windows
    winName = outputFile
    cv.namedWindow(winName, cv.WINDOW_NORMAL)


    # Get classes names
    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    if (not args.image):
        vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M','J','P','G'), 30, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))


    # create network
    net = YoloV3(modelConfiguration, modelWeights, inpWidth, inpHeight, confThreshold, nmsThreshold)


    # Start processing
    while cv.waitKey(1) < 0:
        # get frame from the video
        hasFrame, frame = cap.read()
        
        # Stop the program if reached end of video
        if not hasFrame:
            print("Done processing !!!")
            print("Output file is stored as ", outputFile)
            cv.waitKey(3000)
            break

        # Pass frame to network
        outs = net.predict(frame)

        # Draw bounding boxes on frame
        for out in outs:
            confidence = out[0]
            classId = out[1]
            left = out[2][0]
            top = out[2][1]
            width = out[2][2]
            height = out[2][3]
            drawPred(confidence, classId, left, top, left + width, top + height, classes)


        # Write the frame with the detection boxes
        if (args.image):
            cv.imwrite(outputFile, frame.astype(np.uint8));
        else:
            vid_writer.write(frame.astype(np.uint8))

        cv.imshow(winName, frame)   