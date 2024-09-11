#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <filesystem>
#include <chrono>
#include <tuple>

namespace fs = std::filesystem;

// Global variables
int cellSize_g;
int blockSize_g;
int descriptorSizeDimension_g;
int numBins_g;
int dimofimage_g;

std::vector<std::tuple<std::string, std::string, double, double, double>> memoryUsageLog;

std::string current_filename_g;

void logMemoryUsage(const std::string& label) {
    size_t free_memory, total_memory;
    cudaMemGetInfo(&free_memory, &total_memory);
    size_t used_memory = total_memory - free_memory;

    double usedMemoryMB = used_memory / (1024.0 * 1024.0);
    double freeMemoryMB = free_memory / (1024.0 * 1024.0);
    double totalMemoryMB = total_memory / (1024.0 * 1024.0);

    memoryUsageLog.push_back(std::make_tuple(current_filename_g, label, usedMemoryMB, freeMemoryMB, totalMemoryMB));
}


void saveMemoryUsageLogToCSV(const std::string& filename) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << " for writing." << std::endl;
        return;
    }

    // Write CSV header
    file << "filename,Label,Used Memory (MB),Free Memory (MB),Total Memory (MB)\n";

    // Write each entry from the memoryUsageLog
    for (const auto& entry : memoryUsageLog) {
        file << std::get<0>(entry) << ","  // filename
             << std::get<1>(entry) << ","  // Label
             << std::get<2>(entry) << ","  // Used Memory (MB)
             << std::get<3>(entry) << ","  // Free Memory (MB)
             << std::get<4>(entry) << "\n";  // Total Memory (MB)
    }

    file.close();
}


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
    file << "," << "filename" << "," << "label" <<","<< "ExecTime" <<","<< "DimOfImage" << ","<<"ExecTimeMemLoading"<<"\n";
    file.close();
}


void saveDescriptorAsCSV(const std::vector<double>& descriptor, const std::string& filename,const std::string& path, int label,  double executionTime,  double executionTimeMemory) {
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
    file << "," << path << "," << label << "," << executionTime <<","<< dimofimage_g <<","<< executionTimeMemory << "\n";
    file.close();
}

void computeGradients_seq(const unsigned char* image, int width, int height, std::vector<float>& histograms, int cellSize, int numBins) {
    int numCellsX = width / cellSize;
    int numCellsY = height / cellSize;

    for (int idy = 0; idy < height; ++idy) {
        for (int idx = 0; idx < width; ++idx) {
            float G_x = 0, G_y = 0;

            if (idx > 0 && idx < width - 1) {
                G_x = static_cast<float>(image[idy * width + idx + 1]) - static_cast<float>(image[idy * width + idx - 1]);
            }

            if (idy > 0 && idy < height - 1) {
                G_y = static_cast<float>(image[(idy + 1) * width + idx]) - static_cast<float>(image[(idy - 1) * width + idx]);
            }

            float mag = std::sqrt(G_x * G_x + G_y * G_y);
            float orient = std::atan2(G_y, G_x);

            int cellX = idx / cellSize;
            int cellY = idy / cellSize;
            int histIndex = cellY * numCellsX + cellX;

            float binWidth = M_PI / numBins;

            if (orient < 0) {
                orient += M_PI;
            }

            int bin = static_cast<int>(std::round(orient / binWidth)) % numBins;

            int final_index = histIndex * numBins + bin;
            if (final_index >= histograms.size()) {
                std::cout << "Index out of bounds: " << final_index << std::endl;
            } else {
                histograms[final_index] += mag;
            }
        }
    }
}


__global__ void computeGradients(unsigned char* image, float *d_histograms, int width, int height, int cellSize,float binWidth, int numBins, int histSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int indexCurrent = idy * width + idx;

    if (idx >= width || idy >= height) return; // Boundary check

    float G_x = 0;
    // Compute gradients in x and y directions.
    //The conditional statements check if the current pixel is within the image boundaries.
    //If the pixel is not on the left or right edge of the image, the gradient in the x-direction is computed by subtracting the pixel value on the left from the pixel value on the right.
    if (idx > 0 && idx < width - 1) {
        G_x = (float)image[indexCurrent + 1] - (float)image[indexCurrent - 1];
    }

    float G_y = 0;
    if (idy > 0 && idy < height - 1) {
        G_y = (float)image[(idy + 1) * width + idx] - (float)image[(idy - 1) * width + idx]; // To Do: Capire formula??
    }

    float d_magnitude_var = sqrtf(G_x * G_x + G_y * G_y);
    float d_orientation_var = atan2f(G_y, G_x);

    // Compute histogram bin for the current gradient
    int cellX = idx / cellSize;
    int cellY = idy / cellSize;
    //The division (width / cellSize) calculates the ratio between the width of the grid and the size of each cell.
    //This ratio determines the number of cells that can fit horizontally in the grid.
    //By multiplying this ratio with cellY, we obtain the number of cells that can fit vertically up to the Y-coordinate cellY.
    //Finally, the expression cellY * (width / cellSize) + cellX adds the X-coordinate cellX to the previously calculated value.
    //This addition determines the absolute position of a cell within the grid, considering both its X and Y coordinates.
    int histIndex = cellY * (width / cellSize) + cellX;

    //The following formula calculates the bin index for the current orientation value:
    /*1. `d_orientation[indexCurrent]`: This is a variable or an array element that holds the orientation value at the `indexCurrent` position.
        The orientation value is likely in radians.

    2. `M_PI`: This is a constant defined in the C++ math library that represents the value of pi (π).
        It is used to shift the orientation value by π radians.

    3. `(d_orientation[indexCurrent] + M_PI)`: This expression adds the orientation value to π, effectively shifting the range of values from [-π, π] to [0, 2π].

    4. `binWidth`: This is likely another variable or constant that represents the width of each bin. Bins are used to categorize or group values within a certain range.

    5. `(d_orientation[indexCurrent] + M_PI) / binWidth`: This expression divides the shifted orientation value by the bin width. The result is a floating-point number that represents the bin index.

    6. `floor((d_orientation[indexCurrent] + M_PI) / binWidth)`: The `floor()` function is used to round down the floating-point bin index to the nearest integer. This ensures that the bin index is an integer value.

    */
    // Calculate the gradient orientation as an unsigned angle
    if (d_orientation_var < 0) {
        d_orientation_var += M_PI;  // Normalize to [0, π] range, example if we have -45°, we add 180° to get 135°
    }
    // Calculate the bin index
    int bin = __float2int_rn(d_orientation_var / binWidth) % numBins;

    int final_index = histIndex * numBins + bin;
    if (final_index >= histSize) {
        printf("Index out of bounds: %d\n", final_index);
    }else{
      atomicAdd(&d_histograms[final_index], d_magnitude_var);
    }
}

std::vector<double> computeDescriptorsCUDA(const unsigned char* image, int width, int height, double& executionTime, double& LoadingInMemoryTime) {
    int count;
    cudaError_t error = cudaGetDeviceCount(&count);
    if (error != cudaSuccess || count == 0) {
        std::cerr << "CUDA device initialization failed." << std::endl;
        exit(EXIT_FAILURE);
    }

    int numCellsX = width / cellSize_g;
    int numCellsY = height / cellSize_g;
    cudaDeviceSynchronize();
    logMemoryUsage("Before memory allocation");

    auto startMemory = std::chrono::high_resolution_clock::now();
    
    unsigned char* d_image;
    size_t imageSize = width * height * sizeof(unsigned char); // Size of image in bytes

    // Allocate memory on the device for the image
    cudaError_t status = cudaMalloc(&d_image, imageSize);
    if (status != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }

    // Copy image data from host to device
    status = cudaMemcpy(d_image, image, imageSize, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }

    // Allocate and initialize histograms on the device
    size_t histSize = numCellsX * numCellsY * numBins_g * sizeof(float);
    float* d_histograms;
    status = cudaMalloc((void**)&d_histograms, histSize);
    if (status != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
    status = cudaMemset(d_histograms, 0, histSize);
    if (status != cudaSuccess) {
        fprintf(stderr, "cudaMemset failed: %s\n", cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }

    auto endMemory = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedMemory = endMemory - startMemory;
    LoadingInMemoryTime = elapsedMemory.count();
    cudaDeviceSynchronize();
    logMemoryUsage("After allocating memory");
    
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    float binWidth = M_PI / numBins_g;
    int histSize_vec = numCellsX * numCellsY * numBins_g;

    auto start = std::chrono::high_resolution_clock::now();
    // Launch the CUDA kernel
    computeGradients<<<gridSize, blockSize>>>(d_image, d_histograms, width, height, cellSize_g, binWidth, numBins_g, histSize_vec);
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    executionTime = elapsed.count();

    // Transfer histogram data from device to host
    auto startMemoryFromDevice = std::chrono::high_resolution_clock::now();
    float* h_histograms = new float[numCellsX * numCellsY * numBins_g];
    status = cudaMemcpy(h_histograms, d_histograms, histSize, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }

    auto endMemoryFromDevice = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedMemoryFromDevice = endMemoryFromDevice - startMemoryFromDevice;
    LoadingInMemoryTime += elapsedMemoryFromDevice.count();

    std::vector<double> descriptor;
    for (int i = 0; i <= numCellsY - blockSize_g; ++i) {
        for (int j = 0; j <= numCellsX - blockSize_g; ++j) {
            double blockNorm = 0;
            for (int y = i; y < i + blockSize_g; ++y) {
                for (int x = j; x < j + blockSize_g; ++x) {
                    for (int k = 0; k < numBins_g; ++k) {
                        float histValue = h_histograms[(y * numCellsX + x) * numBins_g + k];
                        blockNorm += histValue * histValue;
                    }
                }
            }
            blockNorm = sqrtf((blockNorm * blockNorm) + 1e-6 * 1e-6);
            for (int y = i; y < i + blockSize_g; ++y) {
                for (int x = j; x < j + blockSize_g; ++x) {
                    for (int k = 0; k < numBins_g; ++k) {
                        double normalizedValue = h_histograms[(y * numCellsX + x) * numBins_g + k] / blockNorm;
                        descriptor.push_back(normalizedValue);
                    }
                }
            }
        }
    }

    cudaFree(d_image);
    delete[] h_histograms;
    cudaFree(d_histograms);
    cudaDeviceSynchronize();
    logMemoryUsage("After freeing all memory");

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
    }


    return descriptor;
}

std::vector<double> computeDescriptorsSeq(const unsigned char* image, int width, int height, double& executionTime) {
    int numCellsX = width / cellSize_g;
    int numCellsY = height / cellSize_g;

    std::vector<float> histograms(numCellsX * numCellsY * numBins_g, 0.0f);
    
    auto start = std::chrono::high_resolution_clock::now();
    computeGradients_seq(image, width, height, histograms, cellSize_g, numBins_g); // Updated to handle raw image data
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    executionTime = elapsed.count();

    std::vector<double> descriptor;
    for (int i = 0; i <= numCellsY - blockSize_g; ++i) {
        for (int j = 0; j <= numCellsX - blockSize_g; ++j) {
            double blockNorm = 0;
            for (int y = i; y < i + blockSize_g; ++y) {
                for (int x = j; x < j + blockSize_g; ++x) {
                    for (int k = 0; k < numBins_g; ++k) {
                        float histValue = histograms[(y * numCellsX + x) * numBins_g + k];
                        blockNorm += histValue * histValue;
                    }
                }
            }

            blockNorm = sqrtf((blockNorm * blockNorm) + 1e-6 * 1e-6);
            for (int y = i; y < i + blockSize_g; ++y) {
                for (int x = j; x < j + blockSize_g; ++x) {
                    for (int k = 0; k < numBins_g; ++k) {
                        double normalizedValue = histograms[(y * numCellsX + x) * numBins_g + k] / blockNorm;
                        descriptor.push_back(normalizedValue);
                    }
                }
            }
        }
    }

    return descriptor;
}


std::vector<double> computeDescriptors(const std::string& image_path, double& executionTime, double& LoadingInMemoryTime, bool cudaAccelerated = true) {
    // Load image with stb_image (load as grayscale)
    int width, height, channels;
    unsigned char* imageBeforeResize = stbi_load(image_path.c_str(), &width, &height, &channels, 1); // Load as grayscale
    if (!imageBeforeResize) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return std::vector<double>();
    }

    // Allocate memory for resized image
    unsigned char* resizedImage = new unsigned char[dimofimage_g * dimofimage_g];

    // Resize image using stb_image_resize
    if (!stbir_resize_uint8_linear(imageBeforeResize, width, height, 0, resizedImage, dimofimage_g, dimofimage_g, 0, STBIR_1CHANNEL)) {        
        std::cerr << "Failed to resize image." << std::endl;
        stbi_image_free(imageBeforeResize);
        delete[] resizedImage;
        return std::vector<double>();
    }

    std::vector<double> descriptor;

    // Call either the CUDA or Sequential descriptor computation
    if (cudaAccelerated) {
        descriptor = computeDescriptorsCUDA(resizedImage, dimofimage_g, dimofimage_g, executionTime, LoadingInMemoryTime);
    } else {
        descriptor = computeDescriptorsSeq(resizedImage, dimofimage_g, dimofimage_g, executionTime);
    }

    // Cleanup
    stbi_image_free(imageBeforeResize);
    delete[] resizedImage;

    std::cout << "End COMPUTE DESCRIPTOR ----------" << std::endl;

    return descriptor;
}


int main(int argc, char** argv) {
     if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <cellSize> <blockSize> <numBins> <outputFile> <dimofimage>" << std::endl;
        return 1;
    }

    cellSize_g = std::stoi(argv[1]);
    blockSize_g = std::stoi(argv[2]);
    numBins_g = std::stoi(argv[3]);
    std::string outputFile = argv[4];
    dimofimage_g = std::stoi(argv[5]);//224

    // Calcute numbers of cells in x and y direction
    int numCellsX = dimofimage_g / cellSize_g;
    int numCellsY = dimofimage_g / cellSize_g;
    /*This is how we calculate the descriptorSizeDimension:

    1. `(numCellsY - blockSize + 1)` calculates the number of blocks in the Y direction.
        Here, `numCellsY` represents the total number of cells in the Y direction, and `blockSize` represents the size of each block.
        By subtracting `blockSize - 1` from `numCellsY`, we account for the overlapping blocks.

    2. `(numCellsX - blockSize + 1)` calculates the number of blocks in the X direction.
        Similar to the previous step, `numCellsX` represents the total number of cells in the X direction, and `blockSize` represents the size of each block.
        Again, we subtract `blockSize - 1` to account for the overlapping blocks.

    3. `blockSize * blockSize` calculates the number of cells within each block.
        Since the blocks are square, we multiply the `blockSize` by itself to get the total number of cells in a block.

    4. `numBins` represents the number of bins used for the descriptor. Each cell in the histogram contains `numBins` values.
        By multiplying all these values together, we get the total size of the descriptor.
        The descriptor size is the product of the number of blocks in the X and Y directions, the number of cells within each block, and the number of bins.
    */
    descriptorSizeDimension_g = (numCellsY - blockSize_g + 1) * (numCellsX - blockSize_g + 1) * blockSize_g * blockSize_g * numBins_g;

    std::string folder_path = "human detection dataset/1"; // Change this to your folder path
    std::vector<int> header;
    for (int i=1; i <= descriptorSizeDimension_g; ++i){
      header.push_back(i);
    }
    std::cout << "Descriptor size : " << descriptorSizeDimension_g << std::endl;
    std::string seq_file = outputFile+"_seq.csv";
    std::string cuda_file = outputFile+"_cuda.csv";
    std::cout << seq_file << std::endl;
    std::cout << cuda_file << std::endl;
    saveDescriptorAsCSVHeader(header, seq_file, "label");
    saveDescriptorAsCSVHeader(header, cuda_file, "label");
    header.clear();
    //Iterate on images where a human is present
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        std::string file_path = entry.path().string();
        std::cout << "Processing image: " << file_path << std::endl;

        double executionTimeCuda = 0.0;
        double executionTimeSeq = 0.0;
        double loadingTimeInMemoryCuda = 0.0;
        double loadingTimeInMemorySeq = 0.0;
        std::vector<double> descriptor = computeDescriptors(file_path, executionTimeCuda, loadingTimeInMemoryCuda);
        std::vector<double> descriptor_seq = computeDescriptors(file_path, executionTimeSeq, loadingTimeInMemorySeq, false);

        current_filename_g = file_path;
        if (descriptor_seq.empty()) {
            std::cout << "Vector is empty" << std::endl;
        } else {
            int label = 1;
            std::cout << descriptor_seq.size() << std::endl;
            std::cout << descriptor[0] << std::endl;
            std::cout << descriptor_seq[0] << std::endl;
            saveDescriptorAsCSV(descriptor, cuda_file, file_path, label, executionTimeCuda, loadingTimeInMemoryCuda);
            saveDescriptorAsCSV(descriptor_seq, seq_file, file_path, label, executionTimeSeq, loadingTimeInMemorySeq);
        }
        descriptor.clear();
        descriptor_seq.clear();
    }

      //Iterate on images where a human is NOT present
      folder_path = "human detection dataset/0";
      for (const auto& entry : fs::directory_iterator(folder_path)) {
        std::string file_path = entry.path().string();
        std::cout << "Processing image: " << file_path << std::endl;

        double executionTimeCuda = 0.0;
        double executionTimeSeq = 0.0;
        double loadingTimeInMemoryCuda = 0.0;
        double loadingTimeInMemorySeq = 0.0;
        std::vector<double> descriptor = computeDescriptors(file_path, executionTimeCuda, loadingTimeInMemoryCuda);
        std::vector<double> descriptor_seq = computeDescriptors(file_path, executionTimeSeq, loadingTimeInMemorySeq, false);        //std::vector<double> descriptor_seq;
        current_filename_g = file_path;
        if (descriptor_seq.empty()) {
            std::cout << "Vector is empty" << std::endl;
        } else {
            int label = 0;
            std::cout << descriptor_seq.size() << std::endl;
            saveDescriptorAsCSV(descriptor, cuda_file, file_path, label, executionTimeCuda, loadingTimeInMemoryCuda);
            saveDescriptorAsCSV(descriptor_seq, seq_file, file_path, label, executionTimeSeq, loadingTimeInMemorySeq);
        }
        descriptor.clear();
        descriptor_seq.clear();
    }

    saveMemoryUsageLogToCSV(outputFile+"_memory_usage_cuda_log.csv");

    return 0;
}