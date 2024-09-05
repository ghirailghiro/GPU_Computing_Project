# Introduction

Welcome to the **HOG-CUDA Human Detection** project! This repository implements an efficient parallelized version of the Histogram of Oriented Gradients (HOG) algorithm for human detection using CUDA. The project compares the performance of a CUDA-accelerated gradient computation against a traditional sequential implementation, with a focus on extracting HOG descriptors from images of varying dimensions.

The algorithm is based on the seminal work by Dalal and Triggs, presented in the paper **[Histograms of Oriented Gradients for Human Detection](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf)**, which demonstrates the effectiveness of HOG descriptors in detecting humans in various poses and backgrounds.

The main objectives of this project include:
- Measuring and comparing the execution times between sequential and CUDA implementations across multiple image dimensions (64x64, 128x128, and 256x256).
- Assessing the scalability and performance advantages of CUDA for large image sizes and descriptor dimensions.
- Evaluating GPU memory usage and overall computational efficiency during image processing.

### Dataset
The dataset used for this project is the Kaggle dataset **[Human Detection Dataset](https://www.kaggle.com/datasets/constantinwerner/human-detection-dataset?resource=download)**, which contains a variety of images with and without human figures, useful for testing and evaluating human detection algorithms.

### Google Colab
The entire project, including both sequential and CUDA implementations, is executed in Google Colab. You can run the code in the provided notebook named **`Gradient_Computation.ipynb`**.

Feel free to explore the code, run experiments, and contribute to further optimizations!
