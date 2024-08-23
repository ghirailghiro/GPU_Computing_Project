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
    file << "," << "label" <<","<< "Exec Time" << "\n";
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
    // Compute gradients in x and y directions. 
    //The conditional statements check if the current pixel is within the image boundaries.
    //If the pixel is not on the left or right edge of the image, the gradient in the x-direction is computed by subtracting the pixel value on the left from the pixel value on the right.
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
    //The division (width / cellSize) calculates the ratio between the width of the grid and the size of each cell. 
    //This ratio determines the number of cells that can fit horizontally in the grid. 
    //By multiplying this ratio with cellY, we obtain the number of cells that can fit vertically up to the Y-coordinate cellY.
    //Finally, the expression cellY * (width / cellSize) + cellX adds the X-coordinate cellX to the previously calculated value. 
    //This addition determines the absolute position of a cell within the grid, considering both its X and Y coordinates.
    int histIndex = cellY * (width / cellSize) + cellX;
    int numBins = 9; // Assuming 9 orientation bins
    float binWidth = M_PI / numBins;
    //The following formula calculates the bin index for the current orientation value:
    /*1. `d_orientation[indexCurrent]`: This is a variable or an array element that holds the orientation value at the `indexCurrent` position. 
        The orientation value is likely in radians.

    2. `M_PI`: This is a constant defined in the C++ math library that represents the value of pi (π). 
        It is used to shift the orientation value by π radians.

    3. `(d_orientation[indexCurrent] + M_PI)`: This expression adds the orientation value to π, effectively shifting the range of values from [-π, π] to [0, 2π].

    4. `binWidth`: This is likely another variable or constant that represents the width of each bin. Bins are used to categorize or group values within a certain range.

    5. `(d_orientation[indexCurrent] + M_PI) / binWidth`: This expression divides the shifted orientation value by the bin width. The result is a floating-point number that represents the bin index.

    6. `floor((d_orientation[indexCurrent] + M_PI) / binWidth)`: The `floor()` function is used to round down the floating-point bin index to the nearest integer. This ensures that the bin index is an integer value.

    Overall, this code snippet calculates the bin index for a given orientation value by shifting the range of values, dividing by the bin width, and rounding down to the nearest integer. 
    The bin index is commonly used in histogram calculations or other applications where values need to be grouped into bins or categories.
    */
    int bin = floor((d_orientation[indexCurrent] + M_PI) / binWidth);
    if (bin == numBins) bin = 0; // Wrap around

    atomicAdd(&d_histograms[histIndex * numBins + bin], d_magnitude[indexCurrent]);
}

std::vector<float> computeDescriptorsCUDA(const cv::Mat& image, double& executionTime) {
    unsigned char* d_image;
    size_t imageSize = image.total() * image.elemSize();
    cudaMalloc(&d_image, imageSize);
    cudaMemcpy(d_image, image.data, imageSize, cudaMemcpyHostToDevice);
    size_t sizeInBytes = image.total() * sizeof(float);
    float* d_magnitude;
    // Allocate memory for magnitude
    cudaError_t status = cudaMalloc((void **)&d_magnitude, sizeInBytes);
    if (status != cudaSuccess) {
        // Handle error (e.g., printing an error message and exiting)
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
    // Initialize d_magnitude to zero
    status = cudaMemset(d_magnitude, 0, sizeInBytes);
    if (status != cudaSuccess) {
        // Handle error
        fprintf(stderr, "cudaMemset failed: %s\n", cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
    sizeInBytes = image.total() * sizeof(float);
    float* d_orientation;
    status = cudaMalloc((void **)&d_orientation, sizeInBytes);
    // Allocate memory for orientation
    if (status != cudaSuccess) {
        // Handle error (e.g., printing an error message and exiting)
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
    // Initialize d_orientation to zero
    status = cudaMemset(d_orientation, 0, sizeInBytes);
    if (status != cudaSuccess) {
        // Handle error
        fprintf(stderr, "cudaMemset failed: %s\n", cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }

    // Assuming image dimensions are reasonable a blocksize 16x16
    //By dividing the image dimensions by the block size and rounding up to the nearest integer, the grid size is determined. 
    //The -1 in the calculation is used to handle cases where the image dimensions are not evenly divisible by the block size. 
    //This ensures that any remaining pixels are included in the grid.
    dim3 blockSize(16, 16);
    dim3 gridSize((image.cols + blockSize.x - 1) / blockSize.x,
                  (image.rows + blockSize.y - 1) / blockSize.y);

    // Allocate memory for histograms
    int cellSize = 64;
    int numCellsX = image.cols / cellSize;
    int numCellsY = image.rows / cellSize;

    // hist size is the number of cells in the x and y direction times 9 bins per cell
    size_t histSize = numCellsX * numCellsY * 9 * sizeof(float);
    float* d_histograms;
    // Allocate memory for histograms
    status = cudaMalloc((void **)&d_histograms, histSize);
    if (status != cudaSuccess) {
        // Handle error
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
    // Initialize histograms to zero
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
    // Normalization of histograms using the sum of squares
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
    
    int cellSize = 64;
    int numCellsX = image.cols / cellSize;
    int numCellsY = image.rows / cellSize;

    std::vector<float> magnitude, orientation;
    // Allocate memory for histograms
    std::vector<float> histograms(numCellsX * numCellsY * 9, 0.0f);
    auto start = std::chrono::high_resolution_clock::now();
    computeGradients_seq(image, magnitude, orientation, histograms, cellSize, 9);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    executionTime = elapsed.count();

    // Normalization of histograms using the sum of squares
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
    // Calcute numbers of cells in x and y direction
    int numCellsX = dimofimage / cellSize;
    int numCellsY = dimofimage / cellSize;
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
    int descriptorSizeDimension = (numCellsY - blockSize + 1) * (numCellsX - blockSize + 1) * blockSize * blockSize * numBins;

    std::string folder_path = "/content/drive/My Drive/GPU Computing/human detection dataset/1"; // Change this to your folder path
    std::vector<int> header;
    for (int i=1; i <= descriptorSizeDimension; ++i){
      header.push_back(i);
    }
    saveDescriptorAsCSVHeader(header, "descriptor_seq.csv", "label");
    saveDescriptorAsCSVHeader(header, "descriptor_cuda.csv", "label");
    header.clear();
    //Iterate on images where a human is present
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

      //Iterate on images where a human is NOT present
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
