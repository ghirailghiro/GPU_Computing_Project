%%writefile gradient_computation.cu
#include <opencv2/opencv.hpp>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <filesystem>
#include <chrono>

namespace fs = std::filesystem;

void saveDescriptorAsCSVHeader(const std::vector<int>& descriptor, const std::string& filename, const std::string& label) {
    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << " for writing." << std::endl;
        return;
    }

    // Write the descriptor to the CSV file
    for (size_t i = 0; i < descriptor.size(); ++i) {
        file << "x" <<descriptor[i];
        if (i < descriptor.size() - 1) {
            file << ",";
        }
    }
    file << "," << label <<","<< "Exec Time" << "\n";
    file.close();
}


void saveDescriptorAsCSV(const std::vector<float>& descriptor, const std::string& filename, int label,  double executionTime) {
    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << " for writing." << std::endl;
        return;
    }

    // Write the descriptor to the CSV file
    for (size_t i = 0; i < descriptor.size(); ++i) {
        file << descriptor[i];
        if (i < descriptor.size() - 1) {
            file << ",";
        }
    }
    file << "," << label << "," << executionTime << "\n";
    file.close();
}

void computeGradients_seq(const cv::Mat& image, std::vector<float>& magnitude, std::vector<float>& orientation, std::vector<float>& histograms, int cellSize, int numBins) {
    magnitude.clear();
    orientation.clear();
    histograms.clear();
    std::cout << "Entering computeGradients" << std::endl;

    // Assuming image dimensions are reasonable for a grid of threads
    int width = image.cols;
    int height = image.rows;
    int numCellsX = width / cellSize;
    int numCellsY = height / cellSize;
    histograms.resize(numCellsX * numCellsY * numBins, 0); // Initialize histogram vector

    // Compute gradients, magnitude, and orientation
    int countBin = 0;
    int countHist = 0;
    int countHistPos = 0;
    for (int idy = 0; idy < height; ++idy) {
        for (int idx = 0; idx < width; ++idx) {
            float G_x = 0, G_y = 0;
            if (idx > 0 && idx < width - 1) {
                G_x = static_cast<float>(image.at<uchar>(idy, idx + 1)) - static_cast<float>(image.at<uchar>(idy, idx - 1));
            }
            if (idy > 0 && idy < height - 1) {
                G_y = static_cast<float>(image.at<uchar>(idy + 1, idx)) - static_cast<float>(image.at<uchar>(idy - 1, idx));
            }

            float mag = std::sqrt(G_x * G_x + G_y * G_y);
            float orient = std::atan2(G_y, G_x);

            magnitude.push_back(mag);
            orientation.push_back(orient);

            // Compute histogram bin for the current gradient
            int cellX = idx / cellSize;
            int cellY = idy / cellSize;
            int histIndex = cellY * numCellsX + cellX;
                countHistPos++; // Ensure index is within bounds
                float binWidth = M_PI / numBins;
                int bin = std::floor((orient + M_PI) / binWidth);
                if (bin == numBins) bin = 0; // Wrap around
                histograms[histIndex * numBins + bin] += mag;
        }
    }
    std::cerr << "------------Summary of errors-----------" << std::endl;
    std::cerr << "Bin out of range: " << countBin << std::endl;
    std::cerr << "Histogram out of range: " << countHist << std::endl;
    std::cerr << "Histogram Pos: " << countHistPos << std::endl;
    std::cout << "Ending computeGradients" << std::endl;
}

__global__ void computeGradients(unsigned char* image, float *d_magnitude, float *d_orientation, float *d_histograms, int width, int height, int cellSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int indexCurrent = idy * width + idx;

    if (idx >= width || idy >= height) return; // Boundary check

    float G_x = 0;
    if (idx > 0 && idx < width - 1) {
        G_x = (float)image[idy * width + (idx + 1)] - (float)image[idy * width + (idx - 1)];
    }

    float G_y = 0;
    if (idy > 0 && idy < height - 1) {
        G_y = (float)image[(idy + 1) * width + idx] - (float)image[(idy - 1) * width + idx];
    }

    d_magnitude[indexCurrent] = sqrtf(G_x * G_x + G_y * G_y);
    d_orientation[indexCurrent] = atan2f(G_y, G_x);

    // Compute histogram bin for the current gradient
    int cellX = idx / cellSize;
    int cellY = idy / cellSize;
    int histIndex = cellY * (width / cellSize) + cellX;
    int numBins = 9; // Assuming 9 orientation bins
    float binWidth = M_PI / numBins;
    int bin = floor((d_orientation[indexCurrent] + M_PI) / binWidth);
    if (bin == numBins) bin = 0; // Wrap around
    // Debug output

    atomicAdd(&d_histograms[histIndex * numBins + bin], d_magnitude[indexCurrent]);
}

std::vector<float> computeDescriptorsCUDA(const cv::Mat& image, double& executionTime) {
    unsigned char* d_image;
    size_t imageSize = image.total() * image.elemSize();
    cudaMalloc(&d_image, imageSize);
    cudaMemcpy(d_image, image.data, imageSize, cudaMemcpyHostToDevice);
    size_t sizeInBytes = image.total() * sizeof(float);
    float* d_magnitude;
    cudaError_t status = cudaMalloc((void **)&d_magnitude, sizeInBytes);
    if (status != cudaSuccess) {
        // Handle error (e.g., printing an error message and exiting)
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
    status = cudaMemset(d_magnitude, 0, sizeInBytes);
    if (status != cudaSuccess) {
        // Handle error
        fprintf(stderr, "cudaMemset failed: %s\n", cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
    sizeInBytes = image.total() * sizeof(float);
    float* d_orientation;
    status = cudaMalloc((void **)&d_orientation, sizeInBytes);
    if (status != cudaSuccess) {
        // Handle error (e.g., printing an error message and exiting)
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
    status = cudaMemset(d_orientation, 0, sizeInBytes);
    if (status != cudaSuccess) {
        // Handle error
        fprintf(stderr, "cudaMemset failed: %s\n", cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }

    // Assuming image dimensions are reasonable for a grid of threads
    dim3 blockSize(16, 16);
    dim3 gridSize((image.cols + blockSize.x - 1) / blockSize.x,
                  (image.rows + blockSize.y - 1) / blockSize.y);

    // Allocate memory for histograms
    int cellSize = 64;
    int numCellsX = image.cols / cellSize;
    int numCellsY = image.rows / cellSize;
    size_t histSize = numCellsX * numCellsY * 9 * sizeof(float);
    float* d_histograms;
    status = cudaMalloc((void **)&d_histograms, histSize);
    if (status != cudaSuccess) {
        // Handle error
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
    status = cudaMemset(d_histograms, 0, histSize);
    if (status != cudaSuccess) {
        // Handle error
        fprintf(stderr, "cudaMemset failed: %s\n", cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }

    auto start = std::chrono::high_resolution_clock::now();
    // Launch the kernel
    computeGradients<<<gridSize, blockSize>>>(d_image, d_magnitude, d_orientation, d_histograms, image.cols, image.rows, cellSize);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    executionTime = elapsed.count();

    // Transfer histogram data from device to host
    float* h_histograms = new float[numCellsX * numCellsY * 9];
    cudaMemcpy(h_histograms, d_histograms, histSize, cudaMemcpyDeviceToHost);
    // Normalization
    for (int i = 0; i < numCellsX * numCellsY; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < 9; ++j) {
            sum += h_histograms[i * 9 + j] * h_histograms[i * 9 + j];
        }
        sum = sqrtf(sum);
        for (int j = 0; j < 9; ++j) {
            h_histograms[i * 9 + j] /= (sum + 1e-6); // Small constant added to avoid division by zero
        }
    }

    // Block Formation and Descriptor Computation
    std::vector<float> descriptor;
    for (int i = 0; i < numCellsY - 1; ++i) {
        for (int j = 0; j < numCellsX - 1; ++j) {
            // Concatenate histograms of four cells into a block
            for (int y = i; y < i + 2; ++y) {
                for (int x = j; x < j + 2; ++x) {
                    for (int k = 0; k < 9; ++k) {
                        descriptor.push_back(h_histograms[(y * numCellsX + x) * 9 + k]);
                    }
                }
            }
        }
    }

    cudaFree(d_image);
    cudaFree(d_magnitude);
    cudaFree(d_orientation);
    delete[] h_histograms;
    cudaFree(d_histograms);

    return descriptor;
}

std::vector<float> computeDescriptorsSeq(const cv::Mat& image, double& executionTime) {
     // Allocate memory for histograms
    int cellSize = 64;
    int numCellsX = image.cols / cellSize;
    int numCellsY = image.rows / cellSize;

    std::vector<float> magnitude, orientation;
    std::vector<float> histograms(numCellsX * numCellsY * 9, 0.0f);
    auto start = std::chrono::high_resolution_clock::now();
    computeGradients_seq(image, magnitude, orientation, histograms, cellSize, 9);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    executionTime = elapsed.count();

    // Normalization
    for (int i = 0; i < numCellsX * numCellsY; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < 9; ++j) {
            sum += histograms[i * 9 + j] * histograms[i * 9 + j];
        }
        sum = sqrtf(sum);
        for (int j = 0; j < 9; ++j) {
            histograms[i * 9 + j] /= (sum + 1e-6); // Small constant added to avoid division by zero
        }
    }

    // Block Formation and Descriptor Computation
    std::vector<float> descriptor;
    for (int i = 0; i < numCellsY - 1; ++i) {
        for (int j = 0; j < numCellsX - 1; ++j) {
            // Concatenate histograms of four cells into a block
            for (int y = i; y < i + 2; ++y) {
                for (int x = j; x < j + 2; ++x) {
                    for (int k = 0; k < 9; ++k) {
                        descriptor.push_back(histograms[(y * numCellsX + x) * 9 + k]);
                    }
                }
            }
        }
    }

    return descriptor;
}

std::vector<float> computeDescriptors(const std::string& image_path, double& executionTime, bool cudaAccelerated = true) {
        // Example: Load an image using OpenCV
    cv::Mat imageBeforeResize = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    cv::Mat image;  
    cv::resize(imageBeforeResize, image, cv::Size(224, 224)); // Resize to standard size
    if(image.empty()) {
        std::cerr << "Failed to load image." << std::endl;
        return std::vector<float>();
    }
    std::vector<float> descriptor;
    if(cudaAccelerated) {
        descriptor = computeDescriptorsCUDA(image, executionTime);
    } else {
        descriptor = computeDescriptorsSeq(image, executionTime);
    }

    // Compare histograms
    /*for (int i = 0; i < descriptor.size(); ++i) {
        if (std::abs(descriptor[i] - descriptor1[i]) > 1e-5) {
            std::cout << "Difference in histogram at index " << i << std::endl;
            std::cout << "CUDA histogram : " << descriptor[i] << std::endl;
            std::cout << "NOT CUDA histogram : " << descriptor1[i] << std::endl;
        }
    }*/

    return descriptor;
}

int main(int argc, char** argv) {
     if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <cellSize> <blockSize> <numBins> <outputFile> <dimofimage>" << std::endl;
        return 1;
    }

    int cellSize = std::stoi(argv[1]);
    int blockSize = std::stoi(argv[2]);
    int numBins = std::stoi(argv[3]);
    std::string outputFile = argv[4];
    int dimofimage = std::stoi(argv[5]);//224
    int numCellsX = dimofimage / cellSize;
    int numCellsY = dimofimage / cellSize;
    int descriptorSizeDimension = (numCellsY - blockSize + 1) * (numCellsX - blockSize + 1) * blockSize * blockSize * numBins;

    std::string folder_path = "/content/drive/My Drive/GPU Computing/human detection dataset/1"; // Change this to your folder path
    std::vector<int> header;
    for (int i=1; i <= descriptorSizeDimension; ++i){
      header.push_back(i);
    }
    saveDescriptorAsCSVHeader(header, "descriptor_seq.csv", "label");
    saveDescriptorAsCSVHeader(header, "descriptor_cuda.csv", "label");
    header.clear();
    //People present class
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        std::string file_path = entry.path().string();
        std::cout << "Processing image: " << file_path << std::endl;

        double executionTimeCuda = 0.0;
        double executionTimeSeq = 0.0;
        std::vector<float> descriptor = computeDescriptors(file_path, executionTimeCuda);
        std::vector<float> descriptor_seq = computeDescriptors(file_path, executionTimeSeq, false);
        if (descriptor.empty() || descriptor_seq.empty()) {
            std::cout << "Vector is empty" << std::endl;
        } else {
            int label = 1;
            saveDescriptorAsCSV(descriptor, "descriptor_cuda.csv", label, executionTimeCuda);
            saveDescriptorAsCSV(descriptor_seq, "descriptor_seq.csv", label, executionTimeSeq);
            descriptor.clear(); // Clear the vector
            descriptor_seq.clear();
        }
    }

      //Not people present class
      folder_path = "/content/drive/My Drive/GPU Computing/human detection dataset/0";
      for (const auto& entry : fs::directory_iterator(folder_path)) {
        std::string file_path = entry.path().string();
        std::cout << "Processing image: " << file_path << std::endl;

        double executionTimeCuda = 0.0;
        double executionTimeSeq = 0.0;
        std::vector<float> descriptor = computeDescriptors(file_path, executionTimeCuda);
        std::vector<float> descriptor_seq = computeDescriptors(file_path, executionTimeSeq, false);
        if (descriptor.empty() || descriptor_seq.empty()) {
            std::cout << "Vector is empty" << std::endl;
        } else {
            int label = 1;
            saveDescriptorAsCSV(descriptor, "descriptor_cuda.csv", label, executionTimeCuda);
            saveDescriptorAsCSV(descriptor_seq, "descriptor_seq.csv", label, executionTimeSeq);
            descriptor.clear(); // Clear the vector
            descriptor_seq.clear();
        }
    }


    return 0;
}
