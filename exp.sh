#!/bin/bash
#g++ ./seq.cpp -o seq `pkg-config --cflags --libs opencv4`
!nvcc gradient_computation.cu -o gradient_computation `pkg-config --cflags --libs opencv4`
# Define the path to the executable
EXECUTABLE="./gradient_computation"

# Define the output directory for the CSV files
OUTPUT_DIR="./output"

# Create the output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Get the current date and time
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")

# Experiment parameters
CELL_SIZES=(64 8)   # Added cell size 8
BLOCK_SIZES=(2 3)   # Added block size 3
NUM_BINS=(9)
OUTPUT_FILE_BASENAME="descriptor"
DIM_IMG=224

# Run experiments
for cell_size in "${CELL_SIZES[@]}"; do
    for block_size in "${BLOCK_SIZES[@]}"; do
        for num_bins in "${NUM_BINS[@]}"; do
            # Construct the output file name
            OUTPUT_FILE="${OUTPUT_DIR}/${OUTPUT_FILE_BASENAME}_cell${cell_size}_block${block_size}_bins${num_bins}_${CURRENT_TIME}.csv"
            
            # Run the experiment
            echo "Running experiment with cell size: $cell_size, block size: $block_size, num bins: $num_bins"
            $EXECUTABLE $cell_size $block_size $num_bins $OUTPUT_FILE $DIM_IMG
        done
    done
done

echo "All experiments completed."

# Define the target directory
TARGET_DIR="/content/drive/My Drive/GPU Computing/output"

# Copy all files from the output directory to the target directory
cp $OUTPUT_DIR/* "$TARGET_DIR"

echo "All files copied to $TARGET_DIR."
echo "All experiments completed."