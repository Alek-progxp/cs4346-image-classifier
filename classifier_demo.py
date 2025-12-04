import time
import image_process
import model_nb_train
import model_perceptron_train

if __name__ == "__main__":
    print("=" * 60)
    print("CLASSIFIER DEMO")
    print("=" * 60)
    
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
    
    # Train and evaluate Naive Bayes model on digit data
    print("\nNaive Bayes on Digit Data:")
    P_y_digit, P_count_given_y_digit, class_counts_digit = model_nb_train.train_nb_whitepx_model(
        digit_train_images, digit_train_labels, 28, 28, bin_size=20
    )
    model_nb_train.print_nb_whitepx_model(P_y_digit, P_count_given_y_digit, class_counts_digit)
    model_nb_train.inspect_P_count_given_y(P_count_given_y_digit, 28, 28, top_k=5)
    
    # Train and evaluate Naive Bayes model on face data
    print("\nNaive Bayes on Face Data:")
    P_y_face, P_count_given_y_face, class_counts_face = model_nb_train.train_nb_whitepx_model(
        face_train_images, face_train_labels, 60, 70, bin_size=20
    )
    model_nb_train.print_nb_whitepx_model(P_y_face, P_count_given_y_face, class_counts_face)
    model_nb_train.inspect_P_count_given_y(P_count_given_y_face, 60, 70, top_k=5)