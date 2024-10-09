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
  - [Implementation Details:](#implementation-details)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Preparation](#data-preparation)
- [Augmentation Techniques](#augmentation-techniques)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Grad-CAM Visualization](#grad-cam-visualization)
- [Jupyter Notebooks](#jupyter-notebooks)
- [Contributing](#contributing)

## Description

This project focuses on processing 10 volleyball match videos captured from different viewpoints. The main objectives are as follows:
 1. Create a 3D reconstruction of the camera positions relative to the field, this point is divided in two steps:
    - intrinsic camera calibration. 
    - extrinsic camera calibration
 2. Develop a tool where you click on the field/on one camera and the same point is visualized on all the other cameras
 3. 3D ball tracking



### 1. Calibration

At first we have calculate the **Intrinsic** and **Extrinsic** parameters.

#### Intrinsic parameter

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


#### Extrinsic Parameters

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


### Implementation Details:

- **Augmentation**: The `apply_augmentations` function is used to generate augmented images from the original input.
- **Meta-Optimization**: The `tune_model` function fine-tunes the model using the augmented images and the specified cost function.
- **Grad-CAM**: The `create_gradcam` function generates heatmaps that visualize the important regions of the input image for the model's predictions.

By using MEMO and MEMO_PLUS, the model becomes more robust to variations and can generalize better to new, unseen data.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/stefanoobonetto/DeepLearning_project.git
   cd DeepLearning_project
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your dataset by placing it in the `datasets` directory.
2. Run the main script to start the training and evaluation process:

   ```bash
   python main.py
   ```

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

## Data Preparation

1. Download the ImageNet-A dataset and place them in the `datasets` directory.
    ```bash
   wget https://people.eecs.berkeley.edu/\~hendrycks/imagenet-a.tar   
   ```
2. Ensure the directory structure is as follows:

   ```
   datasets/
   ├── imagenet-a/
   ```

3. Download the pretrained weights:

   ```bash
   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P weights/
   ```

4. Ensure the directory structure is as follows:

   ```
   weights/
   ├── sam_vit_b_01ec64.pth
   ```

## Augmentation Techniques

This project implements various augmentation techniques to enhance the training data. The available augmentations include:

- Rotation
- Zoom
- Horizontal Flip
- Vertical Flip
- Greyscale
- Inverse
- Blur
- Crop
- Affine
- Change Gamma
- Translation
- Elastic Transform
- Brightness
- Histogram Equalization
- Salt and Pepper Noise
- Gaussian Blur
- Poisson Noise
- Speckle Noise
- Contrast

## Model Training and Evaluation

1. The model is trained using the provided dataset and augmented data.
2. The `tune_model` function fine-tunes the model using MEMO and MEMO_PLUS techniques.
3. The `test_model` function evaluates the model on the test dataset.

## Grad-CAM Visualization

The `create_gradcam` function generates Grad-CAM heatmaps for visualizing the regions of the image that are important for the model's predictions.

## Jupyter Notebooks

This repository also includes a Jupyter Notebook `DeepLearningProject.ipynb` which provides additional insights and interactive analysis. To run the notebook open google colab and select the gpu T4, after that open the file.


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue if you have any suggestions or improvements.


