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
    file << "," << label << "\n";
    file.close();
}


void saveDescriptorAsCSV(const std::vector<float>& descriptor, const std::string& filename, int label) {
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
    file << "," << label << "\n";
    file.close();
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
    atomicAdd(&d_histograms[histIndex * numBins + bin], d_magnitude[indexCurrent]);

    /*if((idx == 2 && idy == 2) || (idx==300 & idy == 300)){
      printf("Value for idx %d e idy %d - bin value: %d - orientation value %f \n", idx, idy, bin,d_orientation[indexCurrent] + M_PI);
    }*/
}

std::vector<float> computeDescriptors(const std::string& image_path, double& executionTime){
        // Example: Load an image using OpenCV
    cv::Mat imageBeforeResize = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    cv::Mat image;
    cv::resize(imageBeforeResize, image, cv::Size(224, 224)); // Resize to standard size
    if(image.empty()) {
        std::cerr << "Failed to load image." << std::endl;
        return std::vector<float>();
    }

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

    /*for (int i = 0; i < numCellsX * numCellsY; ++i) {
        for (int j = 0; j < 9; ++j) {
            std::cout << "Cell " << i << ", Bin " << j << ": " << h_histograms[i * 9 + j] << std::endl;
        }
    }*/
    // Cleanup

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

int main() {
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
    saveDescriptorAsCSVHeader(header, "descriptor.csv", "label");
    header.clear();
    //People present class
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        std::string file_path = entry.path().string();
        std::cout << "Processing image: " << file_path << std::endl;


        std::vector<float> descriptor = computeDescriptors(file_path);
        if (descriptor.empty()) {
            std::cout << "Vector is empty" << std::endl;
        } else {
            int label = 1;
            std::vector<float> descriptor1;
            int length = static_cast<int>(descriptor.size());
            std::cout << "Dimension of descriptor: " << length << std::endl;
            descriptor1.push_back(length);
            saveDescriptorAsCSV(descriptor, "descriptor.csv", label);
            descriptor.clear(); // Clear the vector
            descriptor1.clear();
        }
    }

      //Not people present class
      folder_path = "/content/drive/My Drive/GPU Computing/human detection dataset/0";
      for (const auto& entry : fs::directory_iterator(folder_path)) {
        std::string file_path = entry.path().string();
        std::cout << "Processing image: " << file_path << std::endl;


        std::vector<float> descriptor = computeDescriptors(file_path);
        if (descriptor.empty()) {
            std::cout << "Vector is empty" << std::endl;
        } else {
            int label = 0;
            std::vector<float> descriptor1;
            int length = static_cast<int>(descriptor.size());
            std::cout << "Dimension of descriptor: " << length << std::endl;
            descriptor1.push_back(length);
            saveDescriptorAsCSV(descriptor, "descriptor.csv", label);
            descriptor.clear(); // Clear the vector
            descriptor1.clear();
        }

    }


    return 0;
}
