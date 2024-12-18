import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import wfdb
import matplotlib.pyplot as plt

# Suppress TensorFlow INFO and WARNING logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Function to preprocess ECG data (segmentation)
def preprocess_ecg(record_path, window_size=2, fs=360):
    """
    Preprocess ECG data to create fixed-size segments.
    Args:
        record_path (str): Path to the ECG record file.
        window_size (int): Size of the segment window in seconds.
        fs (int): Sampling frequency (default is 360Hz).
    Returns:
        list: Segments of the ECG signal.
    """
    # Load ECG signal and annotations from the given record
    record = wfdb.rdrecord(record_path)
    ecg_signal = record.p_signal[:, 0]  # Assuming we are using the first signal
    num_samples = len(ecg_signal)

    # Calculate the segment length based on the window size (e.g., 2 seconds)
    segment_length = window_size * fs
    segments = []

    # Create segments by sliding window
    for i in range(0, num_samples, segment_length):
        segment = ecg_signal[i:i + segment_length]
        if len(segment) == segment_length:  # Only keep full segments
            segments.append(segment)

    return segments

# Function to map annotations to segments
def get_labels_from_annotations(record_name, segments, fs=360):
    """
    Get binary labels for each ECG segment based on annotations.
    Args:
        record_name (str): The record name (e.g., '100').
        segments (list): List of ECG signal segments.
        fs (int): The sampling frequency (default is 360Hz).
    Returns:
        list: List of labels (0 = Normal, 1 = Abnormal).
    """
    try:
        annotation = wfdb.rdann(f"data/{record_name}", 'atr')
    except Exception as e:
        print(f"Annotation parsing error for record {record_name}: {e}")
        return np.zeros(len(segments))  # Return all normal labels as fallback

    labels = np.zeros(len(segments))
    for i, symbol in enumerate(annotation.symbol):
        if symbol != 'N':  # Label as abnormal if not 'N' (normal)
            for j, segment in enumerate(segments):
                start_time = j * 2  # Each segment is 2 seconds
                end_time = (j + 1) * 2
                if annotation.sample[i] >= start_time * fs and annotation.sample[i] < end_time * fs:
                    labels[j] = 1  # Assign abnormal label

    return labels

# Function to create the CNN model
def create_model(input_shape):
    """
    Create a CNN model for binary classification.
    Args:
        input_shape (tuple): Shape of the input data (e.g., (720, 1)).
    Returns:
        tf.keras.Model: Compiled CNN model.
    """
    model = Sequential([
        Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape),
        Conv1D(64, kernel_size=3, activation='relu'),
        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification: Normal or Arrhythmia
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Main training script
if __name__ == "__main__":
    record_name = "100"  # Example record name
    segments = preprocess_ecg(f"data/{record_name}", window_size=2)

    # Print segment details for debugging
    print(f"Number of ECG segments: {len(segments)}")
    if len(segments) > 0:
        print(f"Length of one ECG segment: {len(segments[0])}")

    # Get labels for each segment based on annotations
    labels = get_labels_from_annotations(record_name, segments)

    # Analyze label distribution
    normal_count = np.sum(labels == 0)
    abnormal_count = np.sum(labels == 1)
    print(f"Normal Labels: {normal_count}, Abnormal Labels: {abnormal_count}")

    # Prepare data
    X, y = np.array(segments), np.array(labels)

    # Reshape data to (num_samples, 720, 1) for Conv1D
    X = np.reshape(X, (X.shape[0], 720, 1))  # Reshaping to match Conv1D input
    print(f"Shape of X after reshaping: {X.shape}")

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Shape of X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Shape of X_val: {X_val.shape}, y_val: {y_val.shape}")

    # Create and train the model
    model = create_model(input_shape=X_train.shape[1:])
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32)

    # Save the trained model
    model.save("models/trained_model.keras")

    # Evaluate the model
    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

    # Visualize training history
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')
    plt.show()

    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.show()

    # Generate confusion matrix and classification report
    predictions = model.predict(X_val).round().astype(int)
    print(confusion_matrix(y_val, predictions))
    print(classification_report(y_val, predictions))
