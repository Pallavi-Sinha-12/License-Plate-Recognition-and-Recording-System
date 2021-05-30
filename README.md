# License-Plate-Recognition-and-Recording-System
INTRODUCTION TO THE PROJECT-

This project detects and records the license plate number of vehicles. This project involves training of custom yolo model to detect license plate. The camera starts and the real time pictures are clicked and sent into the trained model to detect license plate and crop it and save in a folder. After the camera stops the code runs to read the number from saved images of license plate and prints them. So this project is really helpful in tracking information of vehicles or for parking access control.

ABOUT THE DATASET-

A dataset containing pictures of bikes and cars with license plate visible. The dataset is labelled using open source labelling tool. The labels conatins the coordinates of the detected object.

PREREQUISITE INSTALLATION-

open-cv, numpy, pytesseract

STEPS OF THE PROJECT-

1. DATA COLLECTION- Pictures of vehicles with license plates are collected by clicking images from high resolution camera and from internet.

2. DATA LABELLING- The dataset collected is labelled using open source labelling tool to generate a text file containing coordinates of the object to be detected along with their classes. Both the label files and images are transferred to folder named obj and zipped it.

3. GENERATING PATHS OF THE IMAGES FILES- Made a test and train text files containing paths of images to be trained.

4. PREPARATION OF YOLO CFG FILE- Downloaded yolo-v3-tiny.cfg and edited the file by changing number of classes and number of filters (formula used- (classes_num + 5)*3).

5. PREPARATION OF NAMES AND DATA FILE- Made obj.names file containing names of the class to be detected namely- license_plate. Made obj.data file in which there are informations like number of classes, paths of backup folder, test.txt, train.txt and obj.names file.

6. PREPARATION OF FINAL FOLDER CONTAINING FILES NEEDDED FOR TRAINING- .cfg file, .names file, .data file, obj.zip, test.txt and train.txt are transferred to a folder and uploaded on drive.

7. TRAINED YOLO CUSTOM MODEL ON GOOGLE COLAB- Trained yolo custom model by cloning https://github.com/AlexeyAB/darknet and using the saved folder in step6. The code of training can be found in training.pynb file.

8. DOWNLOADED WEIGHTS FILE- Downloaded weights file of trained model to further use it for inference.

9. CREATION OF FOLDER- Created a folder to further save the cropped license plates of vehicles.

10. WRITING FINAL CODE FOR RUNNING INFERENCE- Written final code to run inferece on real time pictures using .cfg , .names and .weights file. The code is in license_plate1.py. The code when runs clicks pictures of vehicles continuously and send them to trained model to detect the license plate and save the cropped plate to folder created in step 10. After the camera stopped all the cropped pictures are read usig pytesseract. The read numbers are printed.

STEPS TO RUN INFERENCE-

Install the prerequisites mentioned above. Change the paths of yolov3-tiny-obj.cfg, yolov3-tiny-obj_last.weights, obj.names and output folder in license_plate1.py. Open command prompt at this path and write- python license_plate.py The webcam will start and it will start detecting continously unless interrupted using enter key. After closing of webcam it will print the numbers of license plate recorded.

FUTURE SCOPE OF THE PROJECT-

At present this project can be implemented in traffic systems using appropriate hardware. The model is a bit slower on real time if used in mobile devices like raspberry pi. This can be made faster by converting into other faster formats like open vino format(.xml and .bin) or tensorflowlite. But converting this into these forms may lead to less accuracy. So there is always a speed-accuracy trade-off.
