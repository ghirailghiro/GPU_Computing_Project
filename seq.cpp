#include <opencv2/opencv.hpp>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <filesystem>
#include <cmath>
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
        file << "x" << descriptor[i];
        if (i < descriptor.size() - 1) {
            file << ",";
        }
    }
    file << "," << label <<","<< "Exec Time" << "\n";
    file.close();
}

void saveDescriptorAsCSV(std::vector<float> descriptor, int descriptorSize, const std::string& filename, int label, double executionTime) {
    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << " for writing." << std::endl;
        return;
    }

    std::cout << "Saving Descriptor " << std::endl;

    // Write the descriptor to the CSV file
    for (int i = 0; i < descriptorSize; ++i) {
        file << descriptor[i];
        if (i < descriptorSize - 1) {
            file << ",";
        }
    }
    file << "," << label << "," << executionTime << "\n";
    file.close();
    std::cout << "Ending Saving Descriptor " << std::endl;
}

void computeGradients(const cv::Mat& image, std::vector<float>& magnitude, std::vector<float>& orientation, std::vector<float>& histograms, int cellSize, int numBins) {
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
            if (histIndex < (numCellsX * numCellsY)) {
                countHistPos++; // Ensure index is within bounds
                float binWidth = M_PI / numBins;
                int bin = std::floor((orient + M_PI) / binWidth);
                if (bin == numBins) bin = 0; // Wrap around
                if (bin >= 0 && bin < numBins) { // Ensure bin is within bounds
                    histograms[histIndex * numBins + bin] += mag;
                } else {
                    countBin++;
                }
            } else {
                countHist++;
            }

        }
    }
    std::cerr << "------------Summary of errors-----------" << std::endl;
    std::cerr << "Bin out of range: " << countBin << std::endl;
    std::cerr << "Histogram out of range: " << countHist << std::endl;
    std::cerr << "Histogram Pos: " << countHistPos << std::endl;
    std::cout << "Ending computeGradients" << std::endl;
}

std::vector<float> computeDescriptors(const std::string& image_path, int& descriptorSize, int cellSize, int blockSize, int numBins, int dimofimage, int descriptorSizeDimension, double& executionTime) {
    cv::Mat imageBeforeResize = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (imageBeforeResize.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        descriptorSize = 0;
        std::vector<float> descriptorArray;
        return descriptorArray;
    }

    cv::Mat image;
    cv::resize(imageBeforeResize, image, cv::Size(dimofimage, dimofimage)); // Resize to standard size
    int numCellsX = image.cols / cellSize;
    int numCellsY = image.rows / cellSize;
    //int descriptorSizeDimension = (numCellsY - blockSize + 1) * (numCellsX - blockSize + 1) * blockSize * blockSize * numBins;
    //descriptorSize = descriptorSizeDimension;

    //float* descriptorArray = new float[descriptorSizeDimension];
    std::vector<float> magnitude, orientation;
    std::vector<float> histograms(numCellsX * numCellsY * numBins, 0.0f);
    // Timing the computeGradients function
    auto start = std::chrono::high_resolution_clock::now();
    computeGradients(image, magnitude, orientation, histograms, cellSize, numBins);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    executionTime = elapsed.count();
        // Normalization
    for (size_t i = 0; i < histograms.size(); i += numBins) {
        float sum = 0.0f;
        for (int j = 0; j < numBins; ++j) {
            sum += histograms[i + j] * histograms[i + j];
        }
        sum = std::sqrt(sum);
        for (int j = 0; j < numBins; ++j) {
            histograms[i + j] /= (sum + 1e-6); // Small constant added to avoid division by zero
        }
    }

    // Block Formation and Descriptor Computation
    std::vector<float> descriptorArray;
    int index = 0; // Index to keep track of position in the descriptor array
    for (int i = 0; i < numCellsY - blockSize + 1; ++i) {
        for (int j = 0; j < numCellsX - blockSize + 1; ++j) {
            // Concatenate histograms of four cells into a block
            for (int y = i; y < i + blockSize; ++y) {
                for (int x = j; x < j + blockSize; ++x) {
                    for (int k = 0; k < numBins; ++k) {
                        //descriptorArray[index++] = histograms[(y * numCellsX + x) * numBins + k];
                        descriptorArray.push_back(histograms[(y * numCellsX + x) * numBins + k]);
                    }
                }
            }
        }
    }

    return descriptorArray;
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
    std::cout << "Calculate descriptor size dimension: " << descriptorSizeDimension << std::endl;
    std::string folder_path = "human detection dataset/1"; // Change this to your folder path
    std::vector<int> header;
    for (int i = 1; i <= descriptorSizeDimension; ++i) {
        header.push_back(i);
    }
    saveDescriptorAsCSVHeader(header, outputFile, "label");
    header.clear();

    // People present class
    std::string file_path = "human detection dataset/1/0.png";
    std::cout << "Processing image: " << file_path << std::endl;
    int label = 1; // Assuming the label is 1 for the class "People present"
    int descriptorSize = 144;
    double executionTime = 0.0;
    std::vector<float> descriptor = computeDescriptors(file_path, descriptorSize, cellSize, blockSize, numBins, dimofimage, descriptorSizeDimension, executionTime);
    std::cout << "Dimension descriptor: " << descriptorSize << std::endl;
    saveDescriptorAsCSV(descriptor, descriptorSize, outputFile, label, executionTime);
    descriptor.clear(); // Don't forget to deallocate memory

    return 0;
}
