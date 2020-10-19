# organize imports
import cv2
import imutils
import numpy as np
import keyboard
import os
#import tensorflow as tf

#---Control Variables---:
start=True
count=0
font = cv2.FONT_HERSHEY_PLAIN
#########################

#The function for making the background by averaging
bg=None
weight=1
def average(initialFrame):
    global bg
    if start==True:
        bg=initialFrame.copy().astype("float")
        return
    cv2.accumulateWeighted(initialFrame, bg, weight)
####################################################

#This function does the thresholding after 30 frames from start
def threshold(maturedImage):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), maturedImage)
    thresholdedImage = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    return thresholdedImage
####################################################

# Open the device at the ID 0

############################THIS YOLO CNN AND CAMERA INITIALIZAION##############################
net = cv2.dnn.readNet("yolov4-tiny-face_best.weights", "yolov4-tiny-obj.cfg")
classes = [""]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = [[0, 0, 255 ]]
#print(f'colors={colors}')

############################RECORD RELATED VARIABLES##############################
record=False
dataType=1
data1Count=len(os.listdir('./data1'))
data2Count=len(os.listdir('./data2'))
print(f'There are {data1Count} and {data2Count} images')
############################RECORD RELATED VARIABLES##############################


cap = cv2.VideoCapture(0)
#address="http://192.168.0.3:8080/video"
#cap.open(address)
while(True):
    ###################DEALING WITH THE KEY PRESSES STARTS###################
    if keyboard.is_pressed('s'):
        print('s pressed')
        start=True
        count=0
    elif keyboard.is_pressed('r'):
        record=True
    elif keyboard.is_pressed('p'):
        record=False
    elif keyboard.is_pressed('1'):
        dataType=1
    elif keyboard.is_pressed('2'):
        dataType=2
    ###################DEALING WITH THE KEY PRESSES ENDS###################



    #########################FRAME COLLECTION STARTS#####################
    _, frame = cap.read()
    frame=cv2.flip(frame,1)    
    frame=cv2.resize(frame,(650,650))
    img=frame
    height, width, channels = img.shape
    #This img will be used for giving the colored image where yolo will run
    #kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    #img = cv2.filter2D(frame, -1, kernel)
    #########################FRAME COLLECTION ENDS#####################


    
    
    
    
    ###########################DOING THE THRESHING AND YOLOING STARTS#################
    
    
        
    #################Forwarding the image into nn##############################
    #Blob is the processed input image 
    blob = cv2.dnn.blobFromImage(img, 0.00392, (288, 288), (0, 0, 0), True, crop=False)
    #Giving the neural network the blob or processed image
    net.setInput(blob)
    #Getting the feature maps from the output layers
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    #################Forwarding the image into nn done###############################

    ##################Dealing with the outputs of nn and storing in boxes starts#######
    #Taking one feature map at a time
    for out in outs:
        #taking each detection blocks from the feature map
        for detection in out:
            #calculating the confidence score in that detection block
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]                    
            #if the confidence is more than 0.3 then that hand measurements are placed in boxes
            if confidence > 0.3:
                # Object detected                        
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)                                                
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
        #Non max suppression is done here and the index of only those boxes are taken and stored which
        #are left after non max suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            
         ##################Dealing with the outputs of nn and storing in boxes ends#######

        #########################HAND BOUNDING STARTS#############################
        
        
    for i in range(len(boxes)):
        if i in indexes:
            try:
                ######MAKING THE FACE BOX STARTS####################
                x, y, w, h = boxes[i]
                #print("x1,y1 = "+str(x)+", "+str(y))
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)                            
                croppedImg = frame[y:y+h, x:x+w]
                
                
                ######MAKING THE FACE BOX ENDS####################
                
                teethx1=int(x+0.15*w)
                teethy1=int(y+h*0.60)
                teethx2=int(x+0.85*w)
                teethy2=int(y+0.78*h)
                cv2.rectangle(img, (teethx1, teethy1), (teethx2, teethy2), (255,0,0), 1)                            
                croppedImg=img[teethy1:teethy2,teethx1:teethx2]
                croppedImg = cv2.cvtColor(croppedImg, cv2.COLOR_BGR2GRAY)
                croppedImg=cv2.resize(croppedImg,(224,224))
                cv2.imshow('croppedImg',croppedImg)
                
                if (record==True and dataType==1):
                    cv2.imwrite('./data1/'+str(data1Count)+'.jpg',croppedImg)
                    data1Count=data1Count+1
                if (record==True and dataType==2):
                    cv2.imwrite('./data2/'+str(data2Count)+'.jpg',croppedImg)        
                    data2Count=data2Count+1            
            except Exception as e:
                print(e)
            #----BOUNDS FOUND HERE THE SECOND NERUAL NETWORK WILL BE RUNNING-----#

                #cv2.putText(img, label, (x, y + 30), font, 3, color, 2)    
                
        #########################HAND BOUNDING ENDS#############################

        

    ###########################DOING THE THRESHING AND YOLOING  ENDS#################


    if record==True:
        if dataType==1:
            img=cv2.putText(img,f'Record ON data:1 {data1Count}',(20,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0 ),5)
        elif dataType==2:
            img=cv2.putText(img,f'Record ON data:2 {data2Count}',(20,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0 ),5)
    else:
        if dataType==1:
            img=cv2.putText(img,f'Record OFF data:1 {data1Count}',(20,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0 ),5)
        elif dataType==2:
            img=cv2.putText(img,f'Record OFF data:2 {data2Count}',(20,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0 ),5)

    #cv2.imshow('img',img)
    cv2.imshow('frame',frame)       

    #-Loop Breaking Key Check-
    keypress = cv2.waitKey(1) & 0xFF
    if keypress == ord("q"):
        break
    ##########################

cap.release()
cv2.destroyAllWindows()