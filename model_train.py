import numpy as np
import image_process

# Function to train Naive Bayes model using raw pixel values
# Uses a Bernoulli Naive Bayes approach for binary pixel values 
def train_nb_rawpx_model(image_file, label_file, width, height):
    images, labels = image_process.load_ascii_data(image_file, label_file, width, height)

    # Find P(y)
    num_classes = len(set(labels))
    class_counts = np.bincount(labels, minlength=num_classes)
    P_y = class_counts / len(labels)

    # Find P(x|y)
    num_pixels = images.shape[1]
    P_x_given_y = np.zeros((num_classes, num_pixels, 2))  # 2 for binary pixel values
    for c in range(num_classes):
        class_images = images[labels == c]
        pixel_counts = np.sum(class_images, axis=0)
        P_x_given_y[c, :, 1] = (pixel_counts + 1) / (class_counts[c] + 2)  # Laplace smoothing
        P_x_given_y[c, :, 0] = 1 - P_x_given_y[c, :, 1]

    return P_y, P_x_given_y, class_counts

def print_nb_rawpx_model(P_y, P_x_given_y, class_counts):
    print("\nTrained Bernoulli Naive Bayes model (Raw Pixel Features):")
    print("P(y):", P_y)
    print("P(x|y) shape:", P_x_given_y.shape)
    for c in range(len(class_counts)):
        print(f"  Class {c}: count={class_counts[c]}")

# Function to train Naive Bayes model using white pixel counts 
# Reason why I use only white pixel counts is because using both colors would create dependency between features
# (Both colors = Violaton of Naive Bayes assumption of feature independence)
# Using Multinomial Naive Bayes for discrete white pixel counts
def train_nb_whitepx_model(image_file, label_file, width, height, bin_size=20):
    images, labels = image_process.load_ascii_data(image_file, label_file, width, height)
    
    white_pixel_counts = np.sum(images, axis=1).astype(int)

    # Bin the white pixel counts into bins (0-19, 20-39, etc.) to reduce feature space
    # Reduce number of unique count values to improve estimation
    binned_counts = white_pixel_counts // bin_size
    
    # Find P(y)
    num_classes = len(set(labels))
    class_counts = np.bincount(labels, minlength=num_classes)
    P_y = class_counts / len(labels)
    
    # Find P(x|y) for white pixel counts using Multinomial distribution
    # Count frequency of each possible count value per class
    # max_count -> num_bins, is reduced from width*height to (width*height)/bin_size due to binning
    num_bins = (width * height // bin_size) + 1
    P_bin_given_y = np.zeros((num_classes, num_bins + 1))
    
    for c in range(num_classes):
        class_binned_counts = binned_counts[labels == c]
        # Count occurrences of each value with Laplace smoothing
        bin_freq = np.bincount(class_binned_counts, minlength=num_bins + 1)
        # Laplace smoothing: add 1 to all bins
        P_bin_given_y[c, :] = (bin_freq + 1) / (class_counts[c] + num_bins + 1)

    return P_y, P_bin_given_y, class_counts

def print_nb_whitepx_model(P_y, P_count_given_y, class_counts):
    print("\nTrained Multinomial Naive Bayes model (White Pixel Count Feature):")
    print("P(y):", P_y)
    print("P(count|y) shape:", P_count_given_y.shape)
    for c in range(len(class_counts)):
        print(f"  Class {c}: count={class_counts[c]}")
        
# New helper: inspect and validate P_count_given_y
# Implemented to help debug and understand the learned distributions
def inspect_P_count_given_y(P_count_given_y, width, height, top_k=5):
    max_count = (width * height // 20) + 1  # Assuming bin_size=20 as in training function
    print("\nInspecting P(count|y):")
    print(" shape:", P_count_given_y.shape, " expected cols:", max_count + 1)
    # Row sums should be 1 (or very close)
    row_sums = P_count_given_y.sum(axis=1)
    print(" row sums (min, mean, max):", row_sums.min(), row_sums.mean(), row_sums.max())
    if not np.allclose(row_sums, 1.0, atol=1e-8):
        print(" WARNING: some rows do not sum to 1 within tolerance.")
    # Per-class statistics: mean and variance of count under the learned distribution
    counts = np.arange(max_count + 1)
    means = P_count_given_y.dot(counts)
    variances = (P_count_given_y.dot(counts**2) - means**2)
    for c in range(P_count_given_y.shape[0]):
        top_idx = np.argsort(P_count_given_y[c])[::-1][:top_k]
        top_vals = P_count_given_y[c][top_idx]
        print(f" Class {c}: mean={means[c]:.2f}, var={variances[c]:.2f}, \ntop_counts={top_idx.tolist()}, top_probs={np.round(top_vals,6).tolist()}")  

if __name__ == "__main__":
    # Example usage for digit data
    print("Training models for Digit Data")
    digit_image_file = r'cs4346-data\digitdata\trainingimages'
    digit_label_file = r'cs4346-data\digitdata\traininglabels'
    P_y, P_x_given_y, class_counts =  train_nb_rawpx_model(digit_image_file, digit_label_file, 28, 28)
    print_nb_rawpx_model(P_y, P_x_given_y, class_counts)
    P_y, P_count_given_y, class_counts = train_nb_whitepx_model(digit_image_file, digit_label_file, 28, 28)
    print_nb_whitepx_model(P_y, P_count_given_y, class_counts)

    # Inspect the P(count|y) values for digits
    inspect_P_count_given_y(P_count_given_y, 28, 28)
    
    # Example usage for face data
    print("\nTraining models for Face Data")
    face_image_file = r'cs4346-data\facedata\facedatatrain'
    face_label_file = r'cs4346-data\facedata\facedatatrainlabels'
    P_y, P_x_given_y, class_counts = train_nb_rawpx_model(face_image_file, face_label_file, 60, 70)
    print_nb_rawpx_model(P_y, P_x_given_y, class_counts)
    P_y, P_count_given_y, class_counts = train_nb_whitepx_model(face_image_file, face_label_file, 60, 70)
    print_nb_whitepx_model(P_y, P_count_given_y, class_counts)
    
    # Inspect the P(count|y) values for faces
    inspect_P_count_given_y(P_count_given_y, 60, 70)