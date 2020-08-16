# Usage example:  python3 object_detection_yolo.py

import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
from PIL import Image
from PIL import ImageFilter
import pytesseract
k=0

# Initialize the parameters
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4  #Non-maximum suppression threshold

inpWidth = 416  #608     #Width of network's input image
inpHeight = 416 #608     #Height of network's input image

parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
parser.add_argument('--image', help='Path to image file.')
args = parser.parse_args()

image = args.image

# Load names of classes
classesFile = "custom_cfg/voter.names";

classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.

modelConfiguration = "custom_cfg/voter.cfg";
modelWeights = "weights/voter_6000.weights";


net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 1)

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (0, 0, 255),1)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(box, frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        j=0
        #print("out.shape : ", out.shape)
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            
            confidence = scores[classId]
            if detection[4]>confThreshold:
                #print(detection[4], " - ", scores[classId], " - th : ", confThreshold)
                #print(detection[0:4])
                box[j] = list(detection[0:4])
                j=j+1
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
    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        # print('--------------')
        # print(i)
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        right=left+width
        bottom=top+height
  
        drawPred(classIds[i], confidences[i], left, top, right, bottom)
        img1 = Image.open(image)
        img = img1.crop((left, top, right, bottom))
        basewidth = 200
        wpercent = (basewidth/float(img.size[0]))
        hsize = int((float(img.size[1])*float(wpercent)))
        cropped_image = img.resize((basewidth,hsize), Image.ANTIALIAS)
        cropped_image = cropped_image.convert('RGB')
        global k
        cropped_image.save("r" + str(k) + ".jpg", "JPEG",dpi=(300,300))
        img = cv.imread('r'+str(k)+'.jpg')
        k += 1
        #img = cv2.GaussianBlur(img,(5,5),0)
        # img = cv2.medianBlur(img,5) 
        #retval, img = cv2.threshold(img,150,255, cv2.THRESH_BINARY)
        cv.imshow('img',img)
        cv.waitKey(3000)
        file = open("recognized.txt", "a") 
      
        # Apply OCR on the cropped image 
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(img, lang='eng', config=custom_config)
        print(text)
        
        # Appending the text into file 
        file.write(text) 
        file.write("\n") 
        
        # Close the file 
        file.close 

# Process inputs
# Open the image file
if not os.path.isfile(image):
    print("Input image file ", image, " doesn't exist")
    sys.exit(1)
frame = cv.imread(image)
outputFile = image[:-4]+'_yolo_out_py.jpg'

# Create a 4D blob from a frame.
blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

# Sets the input to the network
net.setInput(blob)

# Runs the forward pass to get output of the output layers
outs = net.forward(getOutputsNames(net))

box = {}
# Remove the bounding boxes with low confidence
file = open("recognized.txt", "w+") 
file.write("") 
file.close() 
postprocess(box, frame, outs)

# Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
t, _ = net.getPerfProfile()
label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
#cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

# final_frame = cv.resize(frame, (600, 600))                    # Resize image 
cv.imshow('licence plate detection', frame)

# write image
cv.imwrite(outputFile, frame.astype(np.uint8))

