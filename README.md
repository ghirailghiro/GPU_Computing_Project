# Introduction

Welcome to the **HOG-CUDA Human Detection** repository! This repo implements a parallelized version of the Histogram of Oriented Gradients (HOG) algorithm for human detection using CUDA. This project compares the performance of a CUDA-accelerated gradient computation with a traditional sequential implementation, with a focus on extracting HOG descriptors from images of varying dimensions.

The algorithm is based on the paper published by Dalal and Triggs, presented in **[Histograms of Oriented Gradients for Human Detection](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf)**, which demonstrates the effectiveness of HOG descriptors in detecting humans in various poses and backgrounds.

The main objectives of this project include:
- Measuring and comparing the execution times between sequential and CUDA implementations across multiple image dimensions (64x64, 128x128, and 256x256).
- Assessing the scalability and performance advantages of CUDA for large image sizes and descriptor dimensions.
- Evaluating GPU memory usage and overall computational efficiency during image processing.

### Dataset
The dataset used for this project is the Kaggle dataset **[Human Detection Dataset](https://www.kaggle.com/datasets/constantinwerner/human-detection-dataset?resource=download)**, which contains a series of images with and without human figures, used for testing and evaluating human detection algorithms.

### Google Colab and Personal Computer
The entire project, including both sequential and CUDA implementations, is executed in Google Colab and a Personal PC.

### CUDA Implementation
The CUDA code for both the sequential and CUDA implementations can be found in the file **`gradient_computation.cu`** (for PC execution) and the notebook **`Gradient_Computation.ipynb`** (for colab execution). This file has been created to run the project on a personal machine and it contains the detailed logic used to compute HOG descriptors.

### Running Experiments

The **`exp.sh`** file is a shell script designed to run all the experiments related to the project. It automates the process of testing both the CUDA and sequential implementations across different image dimensions and configurations.

By running **`exp.sh`**, you can:
- Execute the sequential and CUDA HOG descriptor computations.
- Measure execution time and memory usage for different image sizes (64x64, 128x128, and 256x256).
- Collect results for performance comparison.

To run the experiments, execute the script in a terminal:

```bash
./exp.sh
```

This script makes so that all experiments are conducted in a streamlined manner, allowing for easy replication and analysis of the results.

### Contributors
This project was developed by:
- **Michele Ghiradelli**
- **Celeste Nicora**

As part of the final exam for the course GPU Computing at "Università degli Studi di Milano".
