import numpy as np

# Process ASCII image data into numpy arrays, converting characters to binary pixel values and flattening images

# Function to Load ASCII images to an array for processing
def load_ascii_data(image_file, label_file, width, height):
    images = []
    labels = []

    with open(image_file, 'r') as f:
        current_image = []
        for line in f:
            line = line.rstrip('\n')
            if len(line) > 0 or len(current_image) > 0:
                row = [1 if c in ['#', '+'] else 0 for c in line]
                row = (row + [0] * width)[:width]
                current_image.append(row)
            
            if len(current_image) == height:
                images.append(np.array(current_image).flatten())
                current_image = []
    
    with open(label_file, 'r') as f:
        labels = [int(line.strip()) for line in f if line.strip()]
    
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)

# Function to count black and white pixels in images (for 2nd features vector)
def count_bw_pixels(images):
    pixels_per_image = images.shape[1]
    white_pixels = np.sum(images, axis=1)           # Count 1s in each row
    black_pixels = pixels_per_image - white_pixels  # Remaining are 0s
    pixel_counts = np.column_stack((black_pixels, white_pixels))
    return np.array(pixel_counts, dtype=np.int32)

# Function used to output the image data for verification
def print_data(images, labels, width, height):
    img, labels = load_ascii_data(images, labels, width, height)
    print("images shape:", img.shape)
    print("dtype:", img.dtype, "min/max:", img.min(), img.max())
    print("first ten labels:", labels[:10])
    print('\n')

    pixel_counts_digits = count_bw_pixels(img)
    print("Dataset pixel counts shape:", pixel_counts_digits.shape)
    print("\nFirst 5 samples [black, white]:\n", pixel_counts_digits[:5])

if __name__ == "__main__":
    # Verify loading function for digits: 
    # Want 28x28 = 784 pixels per image, datatype = float32, values between 0.0 or 1.0
    digitsrc = r'cs4346-data\digitdata\trainingimages'
    digitsrclb = r'cs4346-data\digitdata\traininglabels'
    print_data(digitsrc, digitsrclb, 28, 28)

    # Verify loading function for faces: 
    # Want 60x70 = 4200 pixels per image, datatype = float32, values between 0.0 or 1.0
    facesrc = r'cs4346-data\facedata\facedatatrain'
    facesrclb = r'cs4346-data\facedata\facedatatrainlabels'
    print_data(facesrc, facesrclb, 60, 70)
