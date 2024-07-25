#include <opencv2/opencv.hpp>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <filesystem>
#include <cmath>

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
    file << "," << label << "\n";
    file.close();
}

void saveDescriptorAsCSV(const float* descriptor, int descriptorSize, const std::string& filename, int label) {
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
    file << "," << label << "\n";
    file.close();
    std::cout << "Ending Saving Descriptor " << std::endl;
}

void computeGradients(const cv::Mat& image, std::vector<float>& magnitude, std::vector<float>& orientation, std::vector<float>& histograms, int cellSize) {
    magnitude.clear();
    orientation.clear();
    histograms.clear();
    std::cout << "Entering computeGradients" << std::endl;

    // Assuming image dimensions are reasonable for a grid of threads
    int width = image.cols;
    int height = image.rows;
    int numCellsX = width / cellSize;
    int numCellsY = height / cellSize;
    int numBins = 9; // Assuming 9 orientation bins
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
            /*std::cout << "Cell X coord: " << cellX << std::endl;
            std::cout << "Cell Y coord: " << cellY << std::endl;
            std::cout << "cellSize coord: " << cellSize << std::endl;
            std::cout << "numCellsX coord: " << numCellsX << std::endl;*/
            int histIndex = cellY * numCellsX + cellX;
            /*std::cout << "histIndex coord: " << histIndex << std::endl;
            std::cout << "histIndex coord: " << histIndex << std::endl;*/
            if (histIndex < (numCellsX * numCellsY)) {
                countHistPos++; // Ensure index is within bounds
                float binWidth = M_PI / numBins;
                int bin = std::floor((orient + M_PI) / binWidth);
                if (bin == numBins) bin = 0; // Wrap around
                if (bin >= 0 && bin < numBins) { // Ensure bin is within bounds
                    histograms[histIndex * numBins + bin] += mag;
                } else {
                    //std::cerr << "Bin index out of range: " << bin << std::endl;
                    countBin++;
                }
            } else {
                //std::cerr << "Histogram index out of range: " << histIndex << std::endl;
                countHist++;
            }

        }
    }
    std::cerr << "------------Summary of errors-----------" << std::endl;
    std::cerr << "Bin out of range: "<< countBin << std::endl;
    std::cerr << "Histogram out of range: "<< countHist << std::endl;
    std::cerr << "Histogram Pos: "<< countHistPos << std::endl;

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
    std::cout << "Ending computeGradients" << std::endl;
}

float* computeDescriptors(const std::string& image_path, int& descriptorSize) {
    cv::Mat imageBeforeResize = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (imageBeforeResize.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        descriptorSize = 0;
        return nullptr;
    }

    cv::Mat image;
    cv::resize(imageBeforeResize, image, cv::Size(224, 224)); // Resize to standard size
    int cellSize = 64;
    int numCellsX = image.cols / cellSize;
    int numCellsY = image.rows / cellSize;
    int numBins = 9; // Assuming 9 orientation bins
    int descriptorSizeDimension = (numCellsY - 1) * (numCellsX - 1) * 4 * numBins; // Size of descriptor array
    descriptorSize = descriptorSizeDimension;

    float* descriptorArray = new float[descriptorSizeDimension];
    std::vector<float> magnitude, orientation;
    std::vector<float> histograms(numCellsX * numCellsY * numBins, 0.0f);
    computeGradients(image, magnitude, orientation, histograms, cellSize);

    // Block Formation and Descriptor Computation
    int index = 0; // Index to keep track of position in the descriptor array
    for (int i = 0; i < numCellsY - 1; ++i) {
        for (int j = 0; j < numCellsX - 1; ++j) {
            // Concatenate histograms of four cells into a block
            for (int y = i; y < i + 2; ++y) {
                for (int x = j; x < j + 2; ++x) {
                    for (int k = 0; k < numBins; ++k) {
                        descriptorArray[index++] = histograms[(y * numCellsX + x) * numBins + k];
                    }
                }
            }
        }
    }

    return descriptorArray;
}

int main() {
    std::string folder_path = "human detection dataset/1"; // Change this to your folder path
    std::vector<int> header;
    for (int i = 1; i <= 144; ++i) {
        header.push_back(i);
    }
    saveDescriptorAsCSVHeader(header, "descriptor.csv", "label");
    header.clear();

    // People present class
    //for (const auto& entry : fs::directory_iterator(folder_path)) {
        //std::string file_path = entry.path().string();
        std::string file_path = "human detection dataset/1/0.png";
        std::cout << "Processing image: " << file_path << std::endl;
        int label = 1; // Assuming the label is 1 for the class "People present"
        int descriptorSize = 144;
        float* descriptor = computeDescriptors(file_path, descriptorSize);
        if (descriptor) {
            std::cout << "Dimension descriptor: " << descriptorSize << std::endl;
            saveDescriptorAsCSV(descriptor, descriptorSize, "descriptor.csv", label);
            delete[] descriptor; // Don't forget to deallocate memory
        } else {
            std::cerr << "Failed to compute descriptors for image: " << file_path << std::endl;
        }
    //}

    return 0;
}
