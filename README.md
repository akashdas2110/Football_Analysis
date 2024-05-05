# Football_Analysis Using YOLOv8

## Introduction:

The goal of this project is to detect and track players, referees, and footballs in a video using YOLO, one of the best AI object detection models available. We will also train the model to improve its performance. Additionally, we will assign players to teams based on the colors of their t-shirts using Kmeans for pixel segmentation and clustering. With this information, we can measure a team's ball acquisition percentage in a match. We will also use optical flow to measure camera movement between frames, enabling us to accurately measure a player's movement. Furthermore, we will implement perspective transformation to represent the scene's depth and perspective, allowing us to measure a player's movement in meters rather than pixels. Finally, we will calculate a player's speed and the distance covered. This project covers various concepts and addresses real-world problems, making it suitable for both beginners and experienced machine learning engineers.

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

* Trained Yolo v5

## Sample video
* Sample input video


## Requirements
To run this project, you need to have the following requirements installed:

* Python 3.x
* ultralytics
* supervision
* OpenCV
* NumPy
* Matplotlib
* Pandas
