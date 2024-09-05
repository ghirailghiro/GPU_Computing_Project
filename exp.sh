# Compile the CUDA code with OpenCV support
echo "Compiling CUDA code..."
nvcc gradient_computation.cu -o gradient_computation `pkg-config --cflags --libs opencv4`
echo "Compilation finished."

# Define the path to the executable
EXECUTABLE="./gradient_computation"

# Define the output directory for the CSV files
OUTPUT_DIR="./output"

# Create the output directory if it doesn't exist
echo "Creating output directory..."
mkdir -p $OUTPUT_DIR
echo "Output directory created: $OUTPUT_DIR"

# Get the current date and time
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
echo "Current timestamp: $CURRENT_TIME"

# Experiment parameters
CELL_SIZES=(64 32 16 8)      # Cell sizes to experiment with
BLOCK_SIZES=(2)
NUM_BINS=(9)              # Number of bins
DIMENSIONS=(64 128 256) # Added dimension list

OUTPUT_FILE_BASENAME="descriptor"

# Run experiments
for dim_img in "${DIMENSIONS[@]}"; do
    for cell_size in "${CELL_SIZES[@]}"; do
        for block_size in "${BLOCK_SIZES[@]}"; do
            for num_bins in "${NUM_BINS[@]}"; do
                # Construct the output file name
                OUTPUT_FILE="${OUTPUT_DIR}/${OUTPUT_FILE_BASENAME}_dim${dim_img}_cell${cell_size}_block${block_size}_bins${num_bins}_${CURRENT_TIME}"

                # Run the experiment
                echo "Running experiment with image dimension: $dim_img, cell size: $cell_size, block size: $block_size, num bins: $num_bins"
                $EXECUTABLE $cell_size $block_size $num_bins $OUTPUT_FILE $dim_img

                echo "Output written to: $OUTPUT_FILE"
            done
        done
    done
done

echo "All experiments completed."

# Define the target directory
TARGET_DIR="/content/drive/My Drive/GPU Computing/output"

# Create the target directory if it doesn't exist
echo "Creating target directory on Google Drive..."
mkdir -p "$TARGET_DIR"
echo "Target directory created: $TARGET_DIR"

# Copy all files from the output directory to the target directory
echo "Copying files to Google Drive..."
cp $OUTPUT_DIR/* "$TARGET_DIR"

echo "All files copied to $TARGET_DIR."
