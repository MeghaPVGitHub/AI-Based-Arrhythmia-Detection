import wfdb
import numpy as np
import scipy.signal as signal
import os

# Define a function to remove noise using a low-pass filter
def remove_noise(ecg_signal, cutoff=50, fs=360):
    """
    Apply a low-pass filter to remove noise from the ECG signal.
    Args:
        ecg_signal (numpy.ndarray): The raw ECG signal.
        cutoff (int): Cutoff frequency in Hz.
        fs (int): Sampling frequency in Hz.
    Returns:
        numpy.ndarray: Denoised ECG signal.
    """
    b, a = signal.butter(4, cutoff / (fs / 2), btype='low')
    return signal.filtfilt(b, a, ecg_signal, axis=0)

# Normalize the ECG signal
def normalize_signal(ecg_signal):
    """
    Normalize the ECG signal to the range [0, 1].
    Args:
        ecg_signal (numpy.ndarray): The raw ECG signal.
    Returns:
        numpy.ndarray: Normalized ECG signal.
    """
    return (ecg_signal - np.min(ecg_signal)) / (np.max(ecg_signal) - np.min(ecg_signal))

# Segment the ECG signal into fixed-length windows
def segment_signal(ecg_signal, window_size, fs=360):
    """
    Segment the ECG signal into fixed-length windows.
    Args:
        ecg_signal (numpy.ndarray): The ECG signal.
        window_size (int): Window size in seconds.
        fs (int): Sampling frequency in Hz.
    Returns:
        list: List of ECG signal segments.
    """
    segment_length = fs * window_size
    return [ecg_signal[i:i + segment_length] for i in range(0, len(ecg_signal), segment_length) if len(ecg_signal[i:i + segment_length]) == segment_length]

# Main preprocessing function
def preprocess_ecg(record_path, window_size=2):
    """
    Preprocess the ECG data: remove noise, normalize, and segment.
    Args:
        record_path (str): Path to the ECG record (without extension).
        window_size (int): Window size in seconds.
    Returns:
        list: Preprocessed ECG segments.
    """
    # Load the ECG signal
    p_signal, fields = wfdb.rdsamp(record_path)
    ecg_signal = p_signal[:, 0]  # Use the first signal (e.g., MLII)

    # Step 1: Remove noise
    denoised_signal = remove_noise(ecg_signal, fs=fields['fs'])

    # Step 2: Normalize
    normalized_signal = normalize_signal(denoised_signal)

    # Step 3: Segment
    segments = segment_signal(normalized_signal, window_size, fs=fields['fs'])

    return segments

# Example usage
if __name__ == "__main__":
    # Set the path to the ECG record
    record_path = "data/100"

    # Preprocess the data
    segments = preprocess_ecg(record_path, window_size=2)

    # Print details of the preprocessed segments
    print(f"Number of segments: {len(segments)}")
    print(f"Segment shape: {segments[0].shape}")
