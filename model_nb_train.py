import numpy as np
import image_process

# Function to train Naive Bayes model using raw pixel values
# Uses a Bernoulli Naive Bayes approach for binary pixel values 
def train_nb_rawpx_model(X, y):
    """
    Train Bernoulli Naive Bayes on raw pixel features.
    
    Args:
        X: feature array, shape (n_samples, num_pixels) with binary values
        y: label array, shape (n_samples,)
    
    Returns:
        P_y: prior probabilities, shape (num_classes,)
        P_x_given_y: conditional probabilities, shape (num_classes, num_pixels, 2)
        class_counts: count of samples per class
    """
    # Find P(y)
    num_classes = len(set(y))
    class_counts = np.bincount(y, minlength=num_classes)
    P_y = class_counts / len(y)

    # Find P(x|y)
    num_pixels = X.shape[1]
    P_x_given_y = np.zeros((num_classes, num_pixels, 2))  # 2 for binary pixel values
    for c in range(num_classes):
        class_images = X[y == c]
        pixel_counts = np.sum(class_images, axis=0)
        P_x_given_y[c, :, 1] = (pixel_counts + 1) / (class_counts[c] + 2)  # Laplace smoothing
        P_x_given_y[c, :, 0] = 1 - P_x_given_y[c, :, 1]

    return P_y, P_x_given_y, class_counts


# Function to predict classes using Bernoulli Naive Bayes (raw pixel features).
def predict_nb_rawpx_model(P_y, P_x_given_y, X):
    """
    Predict classes using Bernoulli Naive Bayes (raw pixel features).
    
    Args:
        P_y: prior probabilities, shape (num_classes,)
        P_x_given_y: P(x=1|y), shape (num_classes, num_pixels, 2)
        X: test data, shape (n_samples, num_pixels) with values 0 or 1
    
    Returns:
        predictions: array of predicted class labels, shape (n_samples,)
    """
    num_classes = P_y.shape[0]
    n_samples = X.shape[0]
    predictions = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        x = X[i]
        # Log-likelihoods to prevent underflow
        log_probs = np.log(P_y + 1e-12)
        
        for c in range(num_classes):
            # Compute log P(x|y=c) for each class
            # Log probability of white and black pixels for total log likelihood.
            log_px = x.dot(np.log(P_x_given_y[c, :, 1] + 1e-12)) + \
                     (1 - x).dot(np.log(P_x_given_y[c, :, 0] + 1e-12))
            log_probs[c] += log_px
        
        # Choose class with highest log probability for this sample
        predictions[i] = np.argmax(log_probs)
    
    return predictions

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
def train_nb_whitepx_model(X, y, bin_size=20):
    """
    Train Multinomial Naive Bayes on binned white pixel counts.
    
    Args:
        X: feature array, shape (n_samples, num_pixels) with binary values
        y: label array, shape (n_samples,)
        bin_size: size of bins for discretizing white pixel counts
    
    Returns:
        P_y: prior probabilities, shape (num_classes,)
        P_bin_given_y: conditional probabilities for bins, shape (num_classes, num_bins+1)
        class_counts: count of samples per class
    """
    white_pixel_counts = np.sum(X, axis=1).astype(int)

    # Bin the white pixel counts into bins (0-19, 20-39, etc.) to reduce feature space
    # Reduce number of unique count values to improve estimation
    binned_counts = white_pixel_counts // bin_size
    
    # Find P(y)
    num_classes = len(set(y))
    class_counts = np.bincount(y, minlength=num_classes)
    P_y = class_counts / len(y)
    
    # Find P(x|y) for white pixel counts using Multinomial distribution
    # Count frequency of each possible count value per class
    # max_count -> max_bin (for binning instead of individual counts)
    max_bin = int(np.max(binned_counts))
    P_bin_given_y = np.zeros((num_classes, max_bin + 1))
    
    for c in range(num_classes):
        class_binned_counts = binned_counts[y == c]
        # Count occurrences - use max_bin + 1 to match array size
        bin_freq = np.bincount(class_binned_counts, minlength=max_bin + 1)
        # Laplace smoothing across ALL bins
        P_bin_given_y[c, :] = (bin_freq + 1) / (class_counts[c] + (max_bin + 1))

    return P_y, P_bin_given_y, class_counts


# Function to predict classes using Multinomial Naive Bayes (white pixel count feature).
def predict_nb_whitepx_model(P_y, P_bin_given_y, X, bin_size=20):
    """
    Predict classes using Multinomial Naive Bayes (white pixel count feature).
    
    Args:
        P_y: prior probabilities, shape (num_classes,)
        P_bin_given_y: P(bin|y), shape (num_classes, num_bins)
        X: test data, shape (n_samples, num_pixels) with binary values
        bin_size: binning factor (same as used in training)
    
    Returns:
        predictions: array of predicted class labels, shape (n_samples,)
    """
    # num_classes not needed here because log_probs is computed directly 
    # (Since two fearures are independent, we can compute log probs directly)
    # num_classes = P_y.shape[0]
    n_samples = X.shape[0]
    predictions = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        white_count = int(np.sum(X[i]))
        bin_idx = white_count // bin_size
        bin_idx = min(bin_idx, P_bin_given_y.shape[1] - 1)
        
        log_probs = np.log(P_y + 1e-12)
        log_probs += np.log(P_bin_given_y[:, bin_idx] + 1e-12)
        
        predictions[i] = np.argmax(log_probs)
    
    return predictions

def print_nb_whitepx_model(P_y, P_bin_given_y, class_counts):
    print("\nTrained Multinomial Naive Bayes model (White Pixel Count Feature):")
    print("P(y):", P_y)
    print("P(bin|y) shape:", P_bin_given_y.shape)
    for c in range(len(class_counts)):
        print(f"  Class {c}: count={class_counts[c]}")
        
# New helper: inspect and validate P_bin_given_y
# Implemented to help debug and understand the learned distributions
def inspect_p_bin_given_y(P_bin_given_y, width, height, bin_size=20, top_k=5):
    num_bins = P_bin_given_y.shape[1]
    print("\nInspecting P(bin|y):")
    print(f" shape: {P_bin_given_y.shape}, bin_size: {bin_size}")
    
    # Row sums should be 1 (or very close)
    row_sums = P_bin_given_y.sum(axis=1)
    print(f" row sums (min, mean, max): {row_sums.min():.6f}, {row_sums.mean():.6f}, {row_sums.max():.6f}")
    if not np.allclose(row_sums, 1.0, atol=1e-8):
        print(" WARNING: some rows do not sum to 1 within tolerance.")
    
    # Per-class statistics: mean and variance of bin indices
    bin_indices = np.arange(num_bins)
    means = P_bin_given_y.dot(bin_indices)
    variances = (P_bin_given_y.dot(bin_indices**2) - means**2)
    
    for c in range(P_bin_given_y.shape[0]):
        top_idx = np.argsort(P_bin_given_y[c])[::-1][:top_k]
        top_vals = P_bin_given_y[c][top_idx]
        bin_ranges = [f"[{idx*bin_size}-{(idx+1)*bin_size-1}]" for idx in top_idx]
        print(f" Class {c}: mean_bin={means[c]:.2f} (~{means[c]*bin_size:.0f} pixels), var={variances[c]:.2f}")
        print(f"  top_bins: {bin_ranges}, top_probs: {np.round(top_vals, 6).tolist()}")

if __name__ == "__main__":
    # ========== DIGIT DATA ==========
    print("=" * 60)
    print("DIGIT DATA - NAIVE BAYES MODELS")
    print("=" * 60)
    
    digit_train_images = r'cs4346-data\digitdata\trainingimages'
    digit_train_labels = r'cs4346-data\digitdata\traininglabels'
    digit_test_images = r'cs4346-data\digitdata\testimages'
    digit_test_labels = r'cs4346-data\digitdata\testlabels'

    # Load data: width=28, height=28
    X_digit_train, y_digit_train = image_process.load_ascii_data(
        digit_train_images, digit_train_labels, 28, 28
    )
    X_digit_test, y_digit_test = image_process.load_ascii_data(
        digit_test_images, digit_test_labels, 28, 28
    )
    
    print(f"\nTraining data shape: {X_digit_train.shape}")
    print(f"Test data shape: {X_digit_test.shape}")

    # Model 1: Raw Pixel Features
    print("\n" + "-" * 60)
    print("Model 1: Bernoulli Naive Bayes (Raw Pixels)")
    print("-" * 60)
    P_y_digit_raw, P_x_given_y_digit_raw, class_counts_digit_raw = train_nb_rawpx_model(
        X_digit_train, y_digit_train
    )
    print_nb_rawpx_model(P_y_digit_raw, P_x_given_y_digit_raw, class_counts_digit_raw)
    
    # Predict and evaluate
    y_pred_digit_train_raw = predict_nb_rawpx_model(
        P_y_digit_raw, P_x_given_y_digit_raw, X_digit_train
    )
    y_pred_digit_test_raw = predict_nb_rawpx_model(
        P_y_digit_raw, P_x_given_y_digit_raw, X_digit_test
    )
    train_acc_digit_raw = np.mean(y_pred_digit_train_raw == y_digit_train)
    test_acc_digit_raw = np.mean(y_pred_digit_test_raw == y_digit_test)
    print(f"\nTraining Accuracy: {train_acc_digit_raw:.4f}")
    print(f"Test Accuracy: {test_acc_digit_raw:.4f}")

    # Model 2: Binned White Pixel Counts
    print("\n" + "-" * 60)
    print("Model 2: Multinomial Naive Bayes (Binned Pixels, bin_size=20)")
    print("-" * 60)
    P_y_digit_bin, P_bin_given_y_digit_bin, class_counts_digit_bin = train_nb_whitepx_model(
        X_digit_train, y_digit_train, 20
    )
    print_nb_whitepx_model(P_y_digit_bin, P_bin_given_y_digit_bin, class_counts_digit_bin)
    inspect_p_bin_given_y(P_bin_given_y_digit_bin, 28, 28, 20)
    
    # Predict and evaluate
    y_pred_digit_train_bin = predict_nb_whitepx_model(
        P_y_digit_bin, P_bin_given_y_digit_bin, X_digit_train, bin_size=20
    )
    y_pred_digit_test_bin = predict_nb_whitepx_model(
        P_y_digit_bin, P_bin_given_y_digit_bin, X_digit_test, bin_size=20
    )
    train_acc_digit_bin = np.mean(y_pred_digit_train_bin == y_digit_train)
    test_acc_digit_bin = np.mean(y_pred_digit_test_bin == y_digit_test)
    print(f"\nTraining Accuracy: {train_acc_digit_bin:.4f}")
    print(f"Test Accuracy: {test_acc_digit_bin:.4f}")
    
    # ========== FACE DATA ==========
    print("\n" + "=" * 60)
    print("FACE DATA - NAIVE BAYES MODELS")
    print("=" * 60)
    
    face_train_images = r'cs4346-data\facedata\facedatatrain'
    face_train_labels = r'cs4346-data\facedata\facedatatrainlabels'
    face_test_images = r'cs4346-data\facedata\facedatatest'
    face_test_labels = r'cs4346-data\facedata\facedatatestlabels'

    # Load data: width=60, height=70 (60 columns × 70 rows = 4200 pixels)
    X_face_train, y_face_train = image_process.load_ascii_data(
        face_train_images, face_train_labels, 60, 70
    )
    X_face_test, y_face_test = image_process.load_ascii_data(
        face_test_images, face_test_labels, 60, 70
    )
    
    print(f"\nTraining data shape: {X_face_train.shape}")
    print(f"Test data shape: {X_face_test.shape}")

    # Model 1: Raw Pixel Features
    print("\n" + "-" * 60)
    print("Model 1: Bernoulli Naive Bayes (Raw Pixels)")
    print("-" * 60)
    P_y_face_raw, P_x_given_y_face_raw, class_counts_face_raw = train_nb_rawpx_model(
        X_face_train, y_face_train
    )
    print_nb_rawpx_model(P_y_face_raw, P_x_given_y_face_raw, class_counts_face_raw)
    
    # Predict and evaluate
    y_pred_face_train_raw = predict_nb_rawpx_model(
        P_y_face_raw, P_x_given_y_face_raw, X_face_train
    )
    y_pred_face_test_raw = predict_nb_rawpx_model(
        P_y_face_raw, P_x_given_y_face_raw, X_face_test
    )
    train_acc_face_raw = np.mean(y_pred_face_train_raw == y_face_train)
    test_acc_face_raw = np.mean(y_pred_face_test_raw == y_face_test)
    print(f"\nTraining Accuracy: {train_acc_face_raw:.4f}")
    print(f"Test Accuracy: {test_acc_face_raw:.4f}")
    
    # Model 2: Binned White Pixel Counts
    print("\n" + "-" * 60)
    print("Model 2: Multinomial Naive Bayes (Binned Pixels, bin_size=50)")
    print("-" * 60)
    P_y_face_bin, P_bin_given_y_face_bin, class_counts_face_bin = train_nb_whitepx_model(
        X_face_train, y_face_train, 50
    )
    print_nb_whitepx_model(P_y_face_bin, P_bin_given_y_face_bin, class_counts_face_bin)
    inspect_p_bin_given_y(P_bin_given_y_face_bin, 60, 70, 50)
    
    # Predict and evaluate
    y_pred_face_train_bin = predict_nb_whitepx_model(
        P_y_face_bin, P_bin_given_y_face_bin, X_face_train, bin_size=50
    )
    y_pred_face_test_bin = predict_nb_whitepx_model(
        P_y_face_bin, P_bin_given_y_face_bin, X_face_test, bin_size=50
    )
    train_acc_face_bin = np.mean(y_pred_face_train_bin == y_face_train)
    test_acc_face_bin = np.mean(y_pred_face_test_bin == y_face_test)
    print(f"\nTraining Accuracy: {train_acc_face_bin:.4f}")
    print(f"Test Accuracy: {test_acc_face_bin:.4f}")
    
    # ========== SUMMARY ==========
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nDigit Classification:")
    print(f"  Raw Pixels Model:")
    print(f"    Training Accuracy: {train_acc_digit_raw:.4f}")
    print(f"    Test Accuracy: {test_acc_digit_raw:.4f}")
    print(f"  Binned Pixels Model (bin_size=20):")
    print(f"    Training Accuracy: {train_acc_digit_bin:.4f}")
    print(f"    Test Accuracy: {test_acc_digit_bin:.4f}")
    
    print("\nFace Detection:")
    print(f"  Raw Pixels Model:")
    print(f"    Training Accuracy: {train_acc_face_raw:.4f}")
    print(f"    Test Accuracy: {test_acc_face_raw:.4f}")
    print(f"  Binned Pixels Model (bin_size=50):")
    print(f"    Training Accuracy: {train_acc_face_bin:.4f}")
    print(f"    Test Accuracy: {test_acc_face_bin:.4f}")