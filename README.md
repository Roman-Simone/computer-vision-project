<div align="center">
  <h1 style="border-bottom: none;">Computer Vision project: 3D CAMERA CALIBRATION (geometry and 3D reconstruction)</h1>
  <img src="https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=ffdd54" alt="Python"/>
  <img src="https://img.shields.io/badge/Numpy-013243?style=flat&logo=numpy&logoColor=white" alt="NumPy"/>
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white" alt="OpenCV"/>
</div>


This repository contains the code and resources for a deep learning project focused on image classification and segmentation using various augmentation techniques and Grad-CAM visualization.

## Table of Contents

- [DeepLearning Project](#deeplearning-project)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
    - [MEMO](#memo)
    - [MEMO\_PLUS](#memo_plus)
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

## Introduction

Deep neural networks often suffer from severe performance degradation when tested on images that differ visually from those encountered during training. This degradation is caused by factors such as domain shift, noise, or changes in lighting.

Recent research has focused on domain adaptation techniques to build deep models that can adapt from an annotated source dataset to a target dataset. However, such methods usually require access to downstream training data, which can be challenging to collect.

An alternative approach is **Test-Time Adaptation (TTA)**, which aims to improve the robustness of a pre-trained neural network to a test dataset, potentially by enhancing the network's predictions on one test sample at a time. 


### MEMO
For this project, MEMO was applied to a pretrained Convolutional Neural Network, **ViT-b/16**, using the **ImageNetV2** dataset. This network operates as follows: given a test point $x \in X$, it produces a conditional output distribution $p(y|x; w)$ over a set of classes $Y$, and predicts a label $\hat{y}$ as:

$$
  \hat{y} = M(x | w) = \arg \max_{y \in Y} p(y | x; w) 
$$

<p align="center" text-align="center">
  <img width="75%" src="https://github.com/christiansassi/deep-learning-project/blob/main/assets/img1.jpg?raw=true">
  <br>
  <span><b>Fig. 1</b> MEMO overview</span>
</p>

Let $A = \{a_1,...,a_M\}$ be a set of augmentations (resizing, cropping, color jittering, etc.). Each augmentation $a_i \in A$ can be applied to an input sample $$x$$, resulting in a transformed sample denoted as $a_i(x)$, as shown in the figure. The objective here is to make the model's prediction invariant to those specific transformations.

MEMO starts by applying a set of $B$ augmentation functions sampled from $A$ to $x$. It then calculates the average, or marginal, output distribution $\bar{p}(y | x; w)$ by averaging the conditional output distributions over these augmentations, represented as:

$$ 
  \bar{p}(y | x; w) = \frac{1}{B} \sum_{i=1}^B p(y | a_i(x); w) 
$$

Since the true label $y$ is not available during testing, the objective of Test-Time Adaptation (TTA) is twofold: (i) to ensure that the model's predictions have the same label $y$ across various augmented versions of the test sample, and (i) to increase the confidence in the model's predictions, given that the augmented versions have the same label. To this end, the model is trained to minimize the entropy of the marginal output distribution across augmentations, defined as:

$$ 
  L(w; x) = H(\bar{p}(\cdot | x;w)) = -\sum_{y \in Y} \bar{p}(y | x;w) \log \bar{p}(y | x;w) 
$$

### MEMO_PLUS

We create our version of this implemantation called MEMO_PLUS that  extends the MEMO technique by incorporating additional segmentation masks and processing steps. The masks help the model focus on specific regions of interest in the image, which can further enhance the robustness and performance of the model.

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


