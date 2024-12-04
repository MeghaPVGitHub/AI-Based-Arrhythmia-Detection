import numpy as np
from scipy.signal import find_peaks

# Function to detect R-peaks
def detect_r_peaks(ecg_signal, fs=360):
    """
    Detect R-peaks in the ECG signal using a peak detection algorithm.
    Args:
        ecg_signal (numpy.ndarray): Preprocessed ECG signal.
        fs (int): Sampling frequency in Hz.
    Returns:
        list: Indices of R-peaks in the signal.
    """
    # Find peaks (adjust height and distance for QRS detection)
    peaks, _ = find_peaks(ecg_signal, height=0.5, distance=fs*0.6)  # 0.6s = typical distance between heartbeats
    return peaks

# Function to calculate HRV
def calculate_hrv(r_peaks, fs=360):
    """
    Calculate Heart Rate Variability (HRV) from R-peak indices.
    Args:
        r_peaks (list): Indices of R-peaks.
        fs (int): Sampling frequency in Hz.
    Returns:
        float: HRV (standard deviation of RR intervals).
    """
    # Convert R-peak indices to time (in milliseconds)
    rr_intervals = np.diff(r_peaks) / fs * 1000  # RR intervals in ms
    return np.std(rr_intervals)  # Standard deviation of RR intervals

# Function to extract statistical features
def extract_statistical_features(ecg_segment):
    """
    Extract statistical features (mean, variance, std) from an ECG segment.
    Args:
        ecg_segment (numpy.ndarray): ECG segment.
    Returns:
        dict: Dictionary of statistical features.
    """
    return {
        "mean": np.mean(ecg_segment),
        "variance": np.var(ecg_segment),
        "std_dev": np.std(ecg_segment),
    }

# Main feature extraction function
def extract_features_from_segments(segments, fs=360):
    """
    Extract features from each ECG segment.
    Args:
        segments (list): List of ECG signal segments.
        fs (int): Sampling frequency in Hz.
    Returns:
        list: List of feature dictionaries for each segment.
    """
    features = []
    for segment in segments:
        # Detect R-peaks
        r_peaks = detect_r_peaks(segment, fs)

        # Calculate HRV
        hrv = calculate_hrv(r_peaks, fs) if len(r_peaks) > 1 else 0

        # Extract statistical features
        stats = extract_statistical_features(segment)

        # Combine all features
        features.append({
            "hrv": hrv,
            **stats
        })
    return features

# Example usage
if __name__ == "__main__":
    # Import preprocessing function
    from preprocess_data import preprocess_ecg

    # Preprocess data to get segments
    record_path = "data/100"
    segments = preprocess_ecg(record_path, window_size=2)

    # Extract features from segments
    features = extract_features_from_segments(segments)

    # Print extracted features for the first segment
    print("Features for first segment:", features[0])
