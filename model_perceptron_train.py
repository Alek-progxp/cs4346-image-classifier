import numpy as np
import image_process

# We want to implement Perceptron models as classes for binary and multiclass classification.
# The multiclass perceptron will use one-vs-rest strategy.
class Perceptron:
    """Binary perceptron classifier"""
    # Function to initialize perceptron with learning rate and epochs (every epoch we go through all data)
    # Therefore the percentage of the dataset we train on will have to be manipulated outside this class
    def __init__(self, learning_rate=0.01, num_epochs=100):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = None
        self.bias = None
    
    # Function to train on dataset X with labels y
    def fit(self, X, y):
        """Train perceptron on binary classification task"""
        # Weights match the shape of input features, bias initially zero
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        
        # Epoch loop (Loops over entire dataset)
        for epoch in range(self.num_epochs):
            errors = 0
            for i in range(len(X)):
                # Compute prediction z (Dot product of weights * inputs + bias)
                # Baseline at 0: if z >= 0 predict 1 else 0s
                z = np.dot(X[i], self.weights) + self.bias
                prediction = 1 if z >= 0 else 0
                
                # Update if prediction is wrong
                # Adjust weights and bias based on error (Negative if false positive, Positive if false negative)
                # Total errors tracked for convergence (No errors means we can stop early)
                if prediction != y[i]:
                    error = y[i] - prediction
                    self.weights += self.learning_rate * error * X[i]
                    self.bias += self.learning_rate * error
                    errors += 1
            
            # Periodic logging of errors for monitoring
            if epoch % 20 == 0:
                print(f"  Epoch {epoch}: {errors} errors")
            
            # Convergence check (Stop early if no errors)
            if errors == 0:
                print(f"  Converged at epoch {epoch}")
                break
    
    # Function to predict on new dataset X (Used after training)
    def predict(self, X):
        """Predict on new data"""
        z = np.dot(X, self.weights) + self.bias
        return (z >= 0).astype(int)

class MulticlassPerceptron:
    # Idea with multiclass perceptron is to create one binary perceptron per class,
    # and during prediction choose the class with the highest confidence (Better than using the binary outputs directly)
    """One-vs-Rest multiclass perceptron"""
    # Initialize with number of classes, learning rate, epochs, and a dictionary to hold perceptrons
    def __init__(self, num_classes, learning_rate=0.01, num_epochs=100):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.perceptrons = {}
    
    # Function to create and train a binary perceptron for each class and store in the dictionary
    def fit(self, X, y):
        """Train one binary perceptron per class"""
        for c in range(self.num_classes):
            print(f"Training perceptron for class {c}")
            # Create binary labels: 1 if class c, 0 otherwise
            binary_labels = (y == c).astype(int)
            
            # Train the current perceptron on the given dataset
            perceptron = Perceptron(self.learning_rate, self.num_epochs)
            perceptron.fit(X, binary_labels)
            self.perceptrons[c] = perceptron
    
    # Function to predict class for new dataset X based on highest confidence from each perceptron
    def predict(self, X):
        """Predict class with highest confidence"""
        confidences = np.zeros((len(X), self.num_classes))
        
        # Get confidence scores from each perceptron
        for c in range(self.num_classes):
            scores = np.dot(X, self.perceptrons[c].weights) + self.perceptrons[c].bias
            confidences[:, c] = scores
        
        # Then return class with highest score
        return np.argmax(confidences, axis=1)
    
    # Function to compute accuracy of predictions on dataset X with labels y
    def accuracy(self, X, y):
        """Compute classification accuracy"""
        predictions = self.predict(X)
        return np.mean(predictions == y)

# Function to counts white pixels and bin them into the specified bin size
def bin_white_pixel_counts(images, bin_size=20):
    """Convert raw pixels to binned white pixel counts"""
    white_pixel_counts = np.sum(images, axis=1)
    binned_counts = (white_pixel_counts // bin_size).astype(np.float32)
    return binned_counts.reshape(-1, 1)  # Reshape to (n_samples, 1)

# Main execution block to train and evaluate perceptron models on digit and face datasets
if __name__ == "__main__":
    print("=" * 60)
    print("DIGIT DATA - PERCEPTRON MODELS")
    print("=" * 60)
    
    # Load digit data
    digit_train_images = r'cs4346-data\digitdata\trainingimages'
    digit_train_labels = r'cs4346-data\digitdata\traininglabels'
    digit_test_images = r'cs4346-data\digitdata\testimages'
    digit_test_labels = r'cs4346-data\digitdata\testlabels'
    
    X_digit_train, y_digit_train = image_process.load_ascii_data(
        digit_train_images, digit_train_labels, 28, 28
    )
    X_digit_test, y_digit_test = image_process.load_ascii_data(
        digit_test_images, digit_test_labels, 28, 28
    )
    
    # Model 1: Digit - Raw pixel features
    print("\nModel 1: Raw Pixel Features")
    print(f"Training data shape: {X_digit_train.shape}")
    perceptron_raw = MulticlassPerceptron(num_classes=10, learning_rate=0.01, num_epochs=100)
    perceptron_raw.fit(X_digit_train, y_digit_train)
    
    train_acc = perceptron_raw.accuracy(X_digit_train, y_digit_train)
    test_acc = perceptron_raw.accuracy(X_digit_test, y_digit_test)
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Model 2: Digit - Binned white pixel counts
    print("\nModel 2: Binned White Pixel Counts (bin_size=20)")
    X_digit_train_binned = bin_white_pixel_counts(X_digit_train, bin_size=20)
    X_digit_test_binned = bin_white_pixel_counts(X_digit_test, bin_size=20)
    print(f"Training data shape: {X_digit_train_binned.shape}")
    
    perceptron_binned = MulticlassPerceptron(num_classes=10, learning_rate=0.01, num_epochs=100)
    perceptron_binned.fit(X_digit_train_binned, y_digit_train)
    
    train_acc = perceptron_binned.accuracy(X_digit_train_binned, y_digit_train)
    test_acc = perceptron_binned.accuracy(X_digit_test_binned, y_digit_test)
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    
    print("\n" + "=" * 60)
    print("FACE DATA - PERCEPTRON MODELS")
    print("=" * 60)
    
    # Load face data
    face_train_images = r'cs4346-data\facedata\facedatatrain'
    face_train_labels = r'cs4346-data\facedata\facedatatrainlabels'
    face_test_images = r'cs4346-data\facedata\facedatatest'
    face_test_labels = r'cs4346-data\facedata\facedatatestlabels'
    
    X_face_train, y_face_train = image_process.load_ascii_data(
        face_train_images, face_train_labels, 60, 70
    )
    X_face_test, y_face_test = image_process.load_ascii_data(
        face_test_images, face_test_labels, 60, 70
    )
    
    # Model 1: Face - Raw pixel features
    print("\nModel 1: Raw Pixel Features")
    print(f"Training data shape: {X_face_train.shape}")
    perceptron_raw_face = MulticlassPerceptron(num_classes=2, learning_rate=0.01, num_epochs=100)
    perceptron_raw_face.fit(X_face_train, y_face_train)
    
    train_acc = perceptron_raw_face.accuracy(X_face_train, y_face_train)
    test_acc = perceptron_raw_face.accuracy(X_face_test, y_face_test)
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Model 2: Face - Binned white pixel counts
    print("\nModel 2: Binned White Pixel Counts (bin_size=50)")
    X_face_train_binned = bin_white_pixel_counts(X_face_train, bin_size=20)
    X_face_test_binned = bin_white_pixel_counts(X_face_test, bin_size=20)
    print(f"Training data shape: {X_face_train_binned.shape}")
    
    perceptron_binned_face = MulticlassPerceptron(num_classes=2, learning_rate=0.01, num_epochs=100)
    perceptron_binned_face.fit(X_face_train_binned, y_face_train)
    
    train_acc = perceptron_binned_face.accuracy(X_face_train_binned, y_face_train)
    test_acc = perceptron_binned_face.accuracy(X_face_test_binned, y_face_test)
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")