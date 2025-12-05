import time
import numpy as np
import image_process
import model_nb_train
import model_perceptron_train
from sklearn.model_selection import train_test_split

# Function to get `fraction` of original as a stratified subset (minimize class imbalance issues)
def stratified_subset(X, y, fraction, seed=None):
    if fraction >= 1.0:
        return X, y
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=fraction, stratify=y, random_state=seed
    )
    return X_sub, y_sub


def print_table(title, data, columns):
    """Pretty print a table with given title, data, and column names."""
    print(f"\n{'=' * 100}")
    print(f"{title:^100}")
    print(f"{'=' * 100}")
    
    # Print header
    header = " | ".join(f"{col:^20}" for col in columns)
    print(header)
    print("-" * 100)
    
    # Print rows
    for row in data:
        row_str = " | ".join(f"{str(val):^20}" for val in row)
        print(row_str)
    print(f"{'=' * 100}\n")


if __name__ == "__main__":
    print("\n" + "=" * 100)
    print(" " * 35 + "CLASSIFIER EVALUATION DEMO")
    print("=" * 100)
    
    # Digit data files
    digit_train_images = r'cs4346-data\digitdata\trainingimages'
    digit_train_labels = r'cs4346-data\digitdata\traininglabels'
    digit_test_images = r'cs4346-data\digitdata\testimages'
    digit_test_labels = r'cs4346-data\digitdata\testlabels'
    
    # Face data files
    face_train_images = r'cs4346-data\facedata\facedatatrain'
    face_train_labels = r'cs4346-data\facedata\facedatatrainlabels'
    face_test_images = r'cs4346-data\facedata\facedatatest'
    face_test_labels = r'cs4346-data\facedata\facedatatestlabels'

    # Load all data once
    print("\nLoading datasets...")
    X_digit_train, y_digit_train = image_process.load_ascii_data(
        digit_train_images, digit_train_labels, 28, 28
    )
    X_digit_test, y_digit_test = image_process.load_ascii_data(
        digit_test_images, digit_test_labels, 28, 28
    )
    X_face_train, y_face_train = image_process.load_ascii_data(
        face_train_images, face_train_labels, 60, 70
    )
    X_face_test, y_face_test = image_process.load_ascii_data(
        face_test_images, face_test_labels, 60, 70
    )

    # Fractions of data to use for training
    fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Initialize result storage
    digit_nb_results = []
    digit_perc_results = []
    face_nb_results = []
    face_perc_results = []
    
    print("\n[Training on DIGITS - Naive Bayes]")
    # Train Naive Bayes models on digit dataset
    for frac in fractions:
        X_digit_sub, y_digit_sub = stratified_subset(X_digit_train, y_digit_train, frac, seed=42)
        
        # Train Bernoulli NB on raw pixels
        start_time = time.time()
        P_y_raw, P_x_given_y, _ = model_nb_train.train_nb_rawpx_model(X_digit_sub, y_digit_sub)
        train_time_raw = time.time() - start_time
        y_pred_raw = model_nb_train.predict_nb_rawpx_model(P_y_raw, P_x_given_y, X_digit_test)
        acc_raw = np.mean(y_pred_raw == y_digit_test)
        
        # Train Multinomial NB on binned white counts
        start_time = time.time()
        P_y_bin, P_bin_given_y, _ = model_nb_train.train_nb_whitepx_model(X_digit_sub, y_digit_sub, bin_size=20)
        train_time_bin = time.time() - start_time
        y_pred_bin = model_nb_train.predict_nb_whitepx_model(P_y_bin, P_bin_given_y, X_digit_test, bin_size=20)
        acc_bin = np.mean(y_pred_bin == y_digit_test)
        
        digit_nb_results.append((f"{frac*100:.0f}%", f"{acc_raw:.4f}", f"{train_time_raw:.2f}s", f"{acc_bin:.4f}", f"{train_time_bin:.2f}s"))
        print(f"  {frac*100:.0f}%: NB Raw={acc_raw:.4f}, NB Bin={acc_bin:.4f}")
    
    print("\n[Training on DIGITS - Perceptron]")
    # Train Perceptron models on digit dataset
    for frac in fractions:
        X_digit_sub, y_digit_sub = stratified_subset(X_digit_train, y_digit_train, frac, seed=42)
        
        # Train Perceptron on raw pixels
        start_time = time.time()
        perc_raw = model_perceptron_train.MulticlassPerceptron(num_classes=10, learning_rate=0.01, num_epochs=50)
        perc_raw.fit(X_digit_sub, y_digit_sub)
        train_time_raw = time.time() - start_time
        y_pred_raw = perc_raw.predict(X_digit_test)
        acc_raw = np.mean(y_pred_raw == y_digit_test)
        
        # Train Perceptron on binned white counts
        X_digit_sub_binned = model_perceptron_train.bin_white_pixel_counts(X_digit_sub, bin_size=20)
        X_digit_test_binned = model_perceptron_train.bin_white_pixel_counts(X_digit_test, bin_size=20)
        start_time = time.time()
        perc_binned = model_perceptron_train.MulticlassPerceptron(num_classes=10, learning_rate=0.01, num_epochs=50)
        perc_binned.fit(X_digit_sub_binned, y_digit_sub)
        train_time_bin = time.time() - start_time
        y_pred_bin = perc_binned.predict(X_digit_test_binned)
        acc_bin = np.mean(y_pred_bin == y_digit_test)
        
        digit_perc_results.append((f"{frac*100:.0f}%", f"{acc_raw:.4f}", f"{train_time_raw:.2f}s", f"{acc_bin:.4f}", f"{train_time_bin:.2f}s"))
        print(f"  {frac*100:.0f}%: Perc Raw={acc_raw:.4f}, Perc Bin={acc_bin:.4f}")
    
    print("\n[Training on FACES - Naive Bayes]")
    # Train Naive Bayes models on face dataset
    for frac in fractions:
        X_face_sub, y_face_sub = stratified_subset(X_face_train, y_face_train, frac, seed=42)
        
        # Train Bernoulli NB on raw pixels
        start_time = time.time()
        P_y_raw, P_x_given_y, _ = model_nb_train.train_nb_rawpx_model(X_face_sub, y_face_sub)
        train_time_raw = time.time() - start_time
        y_pred_raw = model_nb_train.predict_nb_rawpx_model(P_y_raw, P_x_given_y, X_face_test)
        acc_raw = np.mean(y_pred_raw == y_face_test)
        
        # Train Multinomial NB on binned white counts
        start_time = time.time()
        P_y_bin, P_bin_given_y, _ = model_nb_train.train_nb_whitepx_model(X_face_sub, y_face_sub, bin_size=20)
        train_time_bin = time.time() - start_time
        y_pred_bin = model_nb_train.predict_nb_whitepx_model(P_y_bin, P_bin_given_y, X_face_test, bin_size=20)
        acc_bin = np.mean(y_pred_bin == y_face_test)
        
        face_nb_results.append((f"{frac*100:.0f}%", f"{acc_raw:.4f}", f"{train_time_raw:.2f}s", f"{acc_bin:.4f}", f"{train_time_bin:.2f}s"))
        print(f"  {frac*100:.0f}%: NB Raw={acc_raw:.4f}, NB Bin={acc_bin:.4f}")
    
    print("\n[Training on FACES - Perceptron]")
    # Train Perceptron models on face dataset
    for frac in fractions:
        X_face_sub, y_face_sub = stratified_subset(X_face_train, y_face_train, frac, seed=42)
        
        # Train Perceptron on raw pixels
        start_time = time.time()
        perc_raw = model_perceptron_train.MulticlassPerceptron(num_classes=2, learning_rate=0.01, num_epochs=50)
        perc_raw.fit(X_face_sub, y_face_sub)
        train_time_raw = time.time() - start_time
        y_pred_raw = perc_raw.predict(X_face_test)
        acc_raw = np.mean(y_pred_raw == y_face_test)
        
        # Train Perceptron on binned white counts
        X_face_sub_binned = model_perceptron_train.bin_white_pixel_counts(X_face_sub, bin_size=20)
        X_face_test_binned = model_perceptron_train.bin_white_pixel_counts(X_face_test, bin_size=20)
        start_time = time.time()
        perc_binned = model_perceptron_train.MulticlassPerceptron(num_classes=2, learning_rate=0.01, num_epochs=50)
        perc_binned.fit(X_face_sub_binned, y_face_sub)
        train_time_bin = time.time() - start_time
        y_pred_bin = perc_binned.predict(X_face_test_binned)
        acc_bin = np.mean(y_pred_bin == y_face_test)
        
        face_perc_results.append((f"{frac*100:.0f}%", f"{acc_raw:.4f}", f"{train_time_raw:.2f}s", f"{acc_bin:.4f}", f"{train_time_bin:.2f}s"))
        print(f"  {frac*100:.0f}%: Perc Raw={acc_raw:.4f}, Perc Bin={acc_bin:.4f}")
    
    # Print formatted result tables
    print_table("DIGIT CLASSIFICATION - NAIVE BAYES", digit_nb_results, ["Fraction", "Raw Acc", "Raw Time", "Binned Acc", "Binned Time"])
    print_table("DIGIT CLASSIFICATION - PERCEPTRON", digit_perc_results, ["Fraction", "Raw Acc", "Raw Time", "Binned Acc", "Binned Time"])
    print_table("FACE CLASSIFICATION - NAIVE BAYES", face_nb_results, ["Fraction", "Raw Acc", "Raw Time", "Binned Acc", "Binned Time"])
    print_table("FACE CLASSIFICATION - PERCEPTRON", face_perc_results, ["Fraction", "Raw Acc", "Raw Time", "Binned Acc", "Binned Time"])
    
