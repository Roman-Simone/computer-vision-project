<div align="center">
  <h1 style="border-bottom: none;">Computer Vision project: 3D CAMERA CALIBRATION (geometry and 3D reconstruction)</h1>
  <img src="https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=ffdd54" alt="Python"/>
  <img src="https://img.shields.io/badge/Numpy-013243?style=flat&logo=numpy&logoColor=white" alt="NumPy"/>
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white" alt="OpenCV"/>
</div>


This repository contains the code and resources for a Computer Vision project focused on 3D Camera Calibration with also an application of homography matrix to find the same point in different camera views.

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Description](#description)
- [1. Calibration](#1-calibration)
  - [Intrinsic parameter](#intrinsic-parameter)
  - [Extrinsic Parameters](#extrinsic-parameters)
- [Homography:](#homography)
- [3D ball tracking](#3d-ball-tracking)
- [Project Structure](#project-structure)

## Description

This project focuses on processing 10 volleyball match videos captured from different viewpoints. The main objectives are as follows:
 1. Create a 3D reconstruction of the camera positions relative to the field, this point is divided in two steps:
    - intrinsic camera calibration. 
    - extrinsic camera calibration
 2. Develop a tool where you click on the field/on one camera and the same point is visualized on all the other cameras
 3. 3D ball tracking

## 1. Calibration

At first we have calculate the **Intrinsic** and **Extrinsic** parameters.

### Intrinsic parameter

Some pinhole cameras introduce significant distortion to images, primarily in the form of radial and tangential distortion. These distortions can be corrected using a calibration process. For this, we use videos containing a chessboard pattern, and by following the [OpenCV tutorial](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html), we can compute the intrinsic camera matrix and mitigate distortion.

To run the calibration, use the following command:

```bash
python3 intrinsic.py
```

This is an example of the result:

<p align="center"> 
  <img src="data/images/distorted/cam_2.png" alt="Distorted Image" width="30%"/> <br> <i>Figure 1: Distorted Image Before Calibration</i> 
</p> 
<p align="center"> 
  <img src="data/images/undistorted/cam_2.png" alt="Undistorted Image" width="30%"/> <br> <i>Figure 2: Undistorted Image After Calibration</i> 
</p> 


### Extrinsic Parameters

To achieve the 3D reconstruction of camera positions relative to the field, we need to find the **extrinsic parameters**. These parameters describe the rigid body motion (rotation and translation) between the camera and the world frame. In order to compute the extrinsic matrix, at least **four paired points** from the camera plane to the real world are required. For this, we use a script that allows us to select points, typically the corners of the basketball or volleyball court.

To run this script, execute the following command:

```bash
python3 selectPoints.py
```
<p align="center"> <img src="data/images/exampleSelectPoints.png" alt="Distorted Image" width="40%"/> <br> <i>Figure 1: Interface to select points.</i> </p>
Once the points are selected, they are saved in a .json file. You can then use this data to calculate the extrinsic parameters by running:

```bash
python3 extrinsic.py
```

This will produce the extrinsic parameters:

<p align="center"> 
  <img src="data/images/exampleExtrinsic.png" alt="Extrinsic Parameters" width="30%"/> <br> <i>Figure 2: Extrinsic parameters.</i> 
</p> 


## 2. Homography:

To develop a tool where you click on the field/on one camera and the same point is visualized on all the other cameras we have need of Homography matrix that represents the transformation of points in an image plane to another image plane. To calculate these run the command:

```bash
python3 homograpphy.py
```

After that you can try the results with the webapp to have good user experience launch:

```bash
python3 app.py
```

This is an example of the user interface:
<p align="center"> 
  <img src="data/images/exampleUserInterface.png" alt="Extrinsic Parameters" width="50%"/> <br> <i>Figure 2: Extrinsic parameters.</i> 
</p> 

## 3. 3D ball tracking

## Project Structure

```
DeepLearning_project/
├── datasets/
│   ├── imagenet-a/
├── scripts/
│   ├── utils.py
│   ├── model.py
│   ├── gradcam.py
│   ├── functions.py
│   ├── augmentations.py
├── weights/
│   ├── sam_vit_b_01ec64.pth
│   ├── weights_model_in_use.pth
├── notebooks/
│   ├── DeepLearningProject.ipynb
├── main.py
├── requirements.txt
├── README.md
```

# Contacts
For any inquiries, feel free to contact:

- Simone Roman - [simone.roman@studenti.unitn.it](mailto:simone.roman@studenti.unitn.it)

- Stefano Bonetto - [stefano.bonetto@studenti.unitn.it](mailto:stefano.bonetto@studenti.unitn.it)

<br>

<div>
    <a href="https://www.unitn.it/">
        <img src="https://ing-gest.disi.unitn.it/wp-content/uploads/2022/11/marchio_disi_bianco_vert_eng-1024x295.png" width="400px">
    </a>
</div>