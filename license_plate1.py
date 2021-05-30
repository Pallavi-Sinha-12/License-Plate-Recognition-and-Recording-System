#!/usr/bin/env python
# coding: utf-8

# In[9]:


import cv2
import numpy as np
import os
import pytesseract
from collections import Counter


# In[10]:


CONFIDENCE = 0.5  #threshold probability for a label
SCORE_THRESHOLD = 0.5  #a threshold used to filter boxes by score.
IOU_THRESHOLD = 0.5  #threshold intersection value for multiple bounding
config_path = "yolov3-tiny-obj.cfg"
weights = "yolov3-tiny-obj_last.weights"
labels = open("obj.names").read().strip().split("\n")
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")


# In[11]:


net = cv2.dnn.readNetFromDarknet(config_path, weights)


# In[14]:


table = str.maketrans({"(":None, ")":None, "{": None, "}": None, "[": None, "]": None, "|":None, ",":None, "=":None})
table = str.maketrans(dict.fromkeys("(){}[]|,="))
def detect():
    cap = cv2.VideoCapture(0)
    count = 0
    folder_path = "Output"  # path of output folder
    while True:
        ret, image = cap.read()
        h,w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416,416), swapRB = True, crop = False)
        net.setInput(blob)
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        layer_outputs = net.forward(ln)
        boxes, confidences, class_ids = [], [], []
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence>CONFIDENCE:
                    box = detection[:4]*np.array([w,h,w,h])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x,y,int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)
        font_scale = 1
        thickness = 1
        if len(idxs)>0:
            for i in idxs.flatten():
                count = count + 1
                img_path = folder_path + "/" + str(count) + ".jpg"
                x,y = boxes[i][0], boxes[i][1]
                w,h = boxes[i][2], boxes[i][3]
                if x>0 and y>0:
                    cropped = image[y:y+h, x:x+w]
                    cv2.imwrite(img_path, cropped)
                color = [int(c) for c in colors[class_ids[i]]]
                cv2.rectangle(image, (x,y), (x+w, y+h), color = color, thickness= thickness)
                text = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"
                (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
                text_offset_x = x
                text_offset_y = y - 5
                box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
                overlay = image.copy()
                cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
                image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale, color=(0, 0, 0), thickness=thickness)
        cv2.imshow("output", image)
        if cv2.waitKey(1)==13:
            break
    cap.release()
    cv2.destroyAllWindows()
    l = []
    for filename in os.listdir(folder_path):
        img_path = folder_path + "/" + filename
        text = pytesseract.image_to_string(img_path)
        if len(text)>=10:
            text = text.translate(table)
            l.append(text)
    c = Counter(l)
    for item,count in c.items():
        if count>=2:
            print(item)
    


# In[15]:


detect()

