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
def train_nb_whitepx_model(image_file, label_file, width, height):
    images, labels = image_process.load_ascii_data(image_file, label_file, width, height)
    
    white_pixel_counts = np.sum(images, axis=1).astype(int)
    
    # Find P(y)
    num_classes = len(set(labels))
    class_counts = np.bincount(labels, minlength=num_classes)
    P_y = class_counts / len(labels)
    
    # Find P(x|y) for white pixel counts using Multinomial distribution
    # Count frequency of each possible count value per class
    max_count = int(width * height)
    P_count_given_y = np.zeros((num_classes, max_count + 1))
    
    for c in range(num_classes):
        class_white_counts = white_pixel_counts[labels == c]
        # Count occurrences of each value with Laplace smoothing
        count_freq = np.bincount(class_white_counts, minlength=max_count + 1)
        # Laplace smoothing: add 1 to all counts (including unseen values)
        P_count_given_y[c, :] = (count_freq + 1) / (class_counts[c] + max_count + 1)

    return P_y, P_count_given_y, class_counts

def print_nb_whitepx_model(P_y, P_count_given_y, class_counts):
    print("\nTrained Multinomial Naive Bayes model (White Pixel Count Feature):")
    print("P(y):", P_y)
    print("P(count|y) shape:", P_count_given_y.shape)
    for c in range(len(class_counts)):
        print(f"  Class {c}: count={class_counts[c]}")
        
    

if __name__ == "__main__":
    # Example usage for digit data
    print("Training models for Digit Data")
    digit_image_file = r'cs4346-data\digitdata\trainingimages'
    digit_label_file = r'cs4346-data\digitdata\traininglabels'
    P_y, P_x_given_y, class_counts =  train_nb_rawpx_model(digit_image_file, digit_label_file, 28, 28)
    print_nb_rawpx_model(P_y, P_x_given_y, class_counts)
    P_y, P_count_given_y, class_counts = train_nb_whitepx_model(digit_image_file, digit_label_file, 28, 28)
    print_nb_whitepx_model(P_y, P_count_given_y, class_counts)

    
    # Example usage for face data
    print("\nTraining models for Face Data")
    face_image_file = r'cs4346-data\facedata\facedatatrain'
    face_label_file = r'cs4346-data\facedata\facedatatrainlabels'
    P_y, P_x_given_y, class_counts = train_nb_rawpx_model(face_image_file, face_label_file, 60, 70)
    print_nb_rawpx_model(P_y, P_x_given_y, class_counts)
    P_y, P_count_given_y, class_counts = train_nb_whitepx_model(face_image_file, face_label_file, 60, 70)
    print_nb_whitepx_model(P_y, P_count_given_y, class_counts)