import numpy as np

# Process ASCII image data into numpy arrays, converting characters to binary pixel values and flattening images

# Function to Load ASCII images to an array for processing
def load_ascii_data(image_file, label_file, width, height):
    """
    Load ASCII image data and labels.
    
    Args:
        image_file: path to ASCII image file
        label_file: path to label file
        width: number of COLUMNS per image (horizontal dimension) - each row has this many pixels
        height: number of ROWS per image (vertical dimension) - this many rows per image
    
    Returns:
        images: numpy array of shape (n_samples, width*height)
        labels: numpy array of shape (n_samples,)
    """
    images = []
    labels = []

    with open(image_file, 'r') as f:
        current_image = []
        for line in f:
            line = line.rstrip('\n')
            if len(line) > 0 or len(current_image) > 0:
                # Convert characters to binary: # and + are 1, space is 0
                row = [1 if c in ['#', '+'] else 0 for c in line]
                # Pad or truncate to width (columns per row)
                row = (row + [0] * width)[:width]
                current_image.append(row)
            
            # When we have a complete image (height rows), save it
            if len(current_image) == height:
                images.append(np.array(current_image).flatten())
                current_image = []
    
    with open(label_file, 'r') as f:
        labels = [int(line.strip()) for line in f if line.strip()]
    
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)

# Function used to output the image data for verification
def print_data(image_file, label_file, width, height):
    """Print statistics about loaded image data."""
    images, labels = load_ascii_data(image_file, label_file, width, height)
    print(f"Images shape: {images.shape}")
    print(f"Expected: ({len(labels)}, {width*height})")
    print(f"Dtype: {images.dtype}, min/max: {images.min()}, {images.max()}")
    print(f"First ten labels: {labels[:10]}")
    print(f"Label distribution: {np.bincount(labels)}")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("VERIFYING IMAGE LOADING")
    print("=" * 60)
    
    # Verify loading function for digits: 
    # Want 28 rows × 28 cols = 784 pixels per image
    print("\n--- DIGIT DATA ---")
    digitsrc = r'cs4346-data\digitdata\trainingimages'
    digitsrclb = r'cs4346-data\digitdata\traininglabels'
    print_data(digitsrc, digitsrclb, 28, 28) 

    # Verify loading function for faces: 
    # Want 70 rows × 60 cols = 4200 pixels per image
    print("--- FACE DATA ---")
    facesrc = r'cs4346-data\facedata\facedatatrain'
    facesrclb = r'cs4346-data\facedata\facedatatrainlabels'
    print_data(facesrc, facesrclb, 60, 70)
