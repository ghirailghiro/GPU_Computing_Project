# Introduction

Welcome to the **HOG-CUDA Human Detection** project! This repository implements an efficient parallelized version of the Histogram of Oriented Gradients (HOG) algorithm for human detection using CUDA. The project compares the performance of a CUDA-accelerated gradient computation against a traditional sequential implementation, with a focus on extracting HOG descriptors from images of varying dimensions.

The main objectives of this project include:
- Measuring and comparing the execution times between sequential and CUDA implementations across multiple image dimensions (64x64, 128x128, and 256x256).
- Assessing the scalability and performance advantages of CUDA for large image sizes and descriptor dimensions.
- Evaluating GPU memory usage and overall computational efficiency during image processing.

### Dataset
The dataset used for this project is the kaggle dataset **[Human Detection Dataset](https://www.kaggle.com/datasets/constantinwerner/human-detection-dataset?resource=download)**, which contains a variety of images with and without human figures, useful for testing and evaluating human detection algorithms.

### Google Colab
The entire project, including both sequential and CUDA implementations, is executed in Google Colab. You can run the code in the provided notebook named **`insertname`** to easily leverage GPU capabilities for parallel processing.

Feel free to explore the code, run experiments, and contribute to further optimizations!
