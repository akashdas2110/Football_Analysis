# Football_Analysis Using YOLOv8

## Introduction:

This project aims to utilize YOLO, a state-of-the-art AI object detection model, to detect and track players, referees, and footballs in video footage. The objective extends to enhancing the model's performance through training. Moreover, we intend to employ Kmeans for pixel segmentation and clustering to assign players to teams based on their t-shirt colors. This segmentation enables us to calculate a team's ball acquisition percentage during a match. Additionally, optical flow analysis will be used to measure camera movement between frames, facilitating precise evaluation of player movement. Incorporating perspective transformation allows for depth and perspective representation, enabling measurement of player movement in meters rather than pixels. Ultimately, we aim to compute player speed and distance covered. This project encompasses diverse concepts and tackles practical challenges, catering to both novice and seasoned machine learning engineers.


![Screenshot (891)](https://github.com/akashdas2110/Football_Analysis_Using_YOLOv8/assets/112683602/4e206c77-88f3-4ff9-a21b-5aed4c8621c8)

![Screenshot (892)](https://github.com/akashdas2110/Football_Analysis_Using_YOLOv8/assets/112683602/376f59d7-4416-4482-b52d-4a12be5d7e9f)

![image](https://github.com/akashdas2110/Football_Analysis_Using_YOLOv8/assets/112683602/91fc73a6-0bf3-4339-a2bf-7ab883d8ed06)






## Modules Used
The following modules are used in this project:

* YOLO: AI object detection model
* Kmeans: Pixel segmentation and clustering to detect t-shirt color
* Optical Flow: Measure camera movement
* Perspective Transformation: Represent scene depth and perspective
* Speed and distance calculation per player

## Trained Models

* [Trained Yolo v8](https://drive.google.com/file/d/1DC2kCygbBWUKheQ_9cFziCsYVSRw6axK/view)

## Sample video
* [Sample input video](https://drive.google.com/file/d/1dvTx3G2iG1Z5Vj2y3gyk_9vjV7y86Y4J/view?usp=sharing)

## Requirements
To run this project, you need to have the following requirements installed:

* Python 3.x
* ultralytics
* supervision
* OpenCV
* NumPy
* Matplotlib
* Pandas
