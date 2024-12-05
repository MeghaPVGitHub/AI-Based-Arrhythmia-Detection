# AI-Based-Arrhythmia-Detection

## Table of Contents
1. [Data Preprocessing](#data-preprocessing)
    - [Loading and Parsing ECG Data](#loading-and-parsing-ecg-data)
    - [Noise Removal with Low-Pass Filtering](#noise-removal-with-low-pass-filtering)
    - [Normalization of the ECG Signal](#normalization-of-the-ecg-signal)
    - [Segmentation of ECG Signals](#segmentation-of-ecg-signals)
2. [Labeling the Data](#labeling-the-data)
    - [Parsing Annotations](#parsing-annotations)
    - [Labeling Segments](#labeling-segments)
3. [Model Building](#model-building)
    - [Neural Network Architecture](#neural-network-architecture)
    - [Model Training](#model-training)
4. [Error Handling](#error-handling)
5. [Evaluation](#evaluation)
6. [CNN Model Architecture Details](#cnn-model-architecture-details)

---

## Data Preprocessing

### Loading and Parsing ECG Data
- **Function**: `wfdb.rdsamp()`
- **Purpose**: Reads ECG signals and associated metadata (e.g., sampling frequency, signal names) from files in WFDB format.
- **Example**:
  ```python
  p_signal, fields = wfdb.rdsamp('data/100')
  ```

### Noise Removal with Low-Pass Filtering
- **Logic**: Removes high-frequency noise (e.g., from muscle contractions or electrical interference) while preserving low-frequency ECG signals.
- **Process**:
  1. **Filter Design**: Uses a Butterworth filter with a cutoff frequency of 50 Hz.
  2. **Signal Filtering**: Applies the filter to the ECG signal using `signal.filtfilt()`.

  ```python
  b, a = signal.butter(4, cutoff / (fs / 2), btype='low')
  filtered_signal = signal.filtfilt(b, a, ecg_signal, axis=0)
  ```

### Normalization of the ECG Signal
- **Logic**: Ensures that ECG signal values fall within a consistent range [0, 1] for easier processing by the model.
- **Formula**:
  ```python
  normalized_signal = (ecg_signal - np.min(ecg_signal)) / (np.max(ecg_signal) - np.min(ecg_signal))
  ```

### Segmentation of ECG Signals
- **Logic**: Splits the ECG signal into fixed-length windows for better model learning.
- **Example**: For a 360 Hz sampling frequency and 2-second windows:
  ```python
  segment_length = fs * window_size
  segments = [ecg_signal[i:i + segment_length] for i in range(0, len(ecg_signal), segment_length) if len(ecg_signal[i:i + segment_length]) == segment_length]
  ```

---

## Labeling the Data

### Parsing Annotations
- **Function**: `wfdb.rdann()`
- **Purpose**: Reads annotation files containing event types (e.g., normal or abnormal beats) for ECG signals.
- **Example**:
  ```python
  annotation = wfdb.rdann('data/100', 'atr')
  ```

### Labeling Segments
- **Logic**: 
  - Segments with abnormal beats (non-'N' annotations) are labeled as `1`.
  - Segments with only normal beats are labeled as `0`.
- **Example**:
  ```python
  if symbol != 'N':  # Non-'N' annotations are considered abnormal
      labels[j] = 1  # Assign abnormal label
  ```

---

## Model Building

### Neural Network Architecture
- **Type**: 1D Convolutional Neural Network (CNN)
- **Purpose**: Classifies ECG segments as normal or abnormal.
- **Layers**:
  - Convolutional layers for feature extraction.
  - Fully connected layers for classification.
  - Output layer with sigmoid activation for binary classification.
- **Example**:
  ```python
  model = Sequential([
      Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape),
      Conv1D(64, kernel_size=3, activation='relu'),
      Flatten(),
      Dense(128, activation='relu'),
      Dropout(0.5),
      Dense(1, activation='sigmoid')
  ])
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  ```

### Model Training
- **Logic**:
  - Loss function: Binary cross-entropy.
  - Metric: Accuracy.
  - Validation: Ensures generalization on unseen data.
- **Example**:
  ```python
  history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32)
  ```

---

## Error Handling
- **Parsing Errors**: 
  - Issue: Errors during annotation parsing (e.g., `OverflowError`).
  - Solution: Skip problematic records or clean annotation files manually.

---

## Evaluation

### Metrics
- **Accuracy**: Percentage of correct predictions.
- **Precision**: Proportion of true positive predictions out of all positive predictions.
- **Recall**: Proportion of actual positive samples correctly predicted.
- **F1-Score**: Harmonic mean of precision and recall.
- **Confusion Matrix**: Visual representation of predictions (normal vs. abnormal).

---

## CNN Model Architecture Details

1. **Conv1D Layer**:
   - Detects local patterns in the ECG signal.
   - **Kernel Size**: 3
   - **Filters**: 32 (first layer), 64 (second layer).

2. **ReLU Activation**:
   - Adds non-linearity for better learning.

3. **Flatten Layer**:
   - Converts 2D data into a 1D vector.

4. **Dense Layer**:
   - **First Dense Layer**: Combines extracted features (128 neurons).
   - **Dropout Layer**: Prevents overfitting.

5. **Output Layer**:
   - **Activation**: Sigmoid for binary classification.

---
