import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from typing import List, Tuple

class ModifiedAnomalyDetector:
    """
    Anomaly detector using Exponentially Weighted Moving Average (EWMA).
    
    This detector maintains a moving window of recent values and uses EWMA
    to detect anomalies based on deviations from the expected value.
    """
    def __init__(self, window_size: int = 50, threshold: float = 2.5, alpha: float = 0.1):
        """
        Initialize the anomaly detector.

        Args:
            window_size (int): Size of the moving window.
            threshold (float): Z-score threshold for anomaly detection.
            alpha (float): EWMA smoothing factor.
        """
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        if not isinstance(threshold, (int, float)) or threshold <= 0:
            raise ValueError("threshold must be a positive number")
        if not isinstance(alpha, float) or alpha <= 0 or alpha >= 1:
            raise ValueError("alpha must be a float between 0 and 1")

        self.window_size = window_size
        self.threshold = threshold
        self.window = deque(maxlen=window_size)
        self.ewma = None
        self.ewmv = None
        self.alpha = alpha

    def update(self, value: float) -> bool:
        """
        Update the detector with a new value and check for anomalies.

        Args:
            value (float): The new data point to be checked.

        Returns:
            bool: True if the value is an anomaly, False otherwise.
        """
        try:
            if not isinstance(value, (int, float)):
                raise ValueError("Input value must be a number")

            self.window.append(value)
            
            if self.ewma is None:
                self.ewma = value
                self.ewmv = 0
                return False

            diff = value - self.ewma
            incr = self.alpha * diff
            self.ewma += incr
            self.ewmv = (1 - self.alpha) * (self.ewmv + self.alpha * diff ** 2)
            
            if len(self.window) < self.window_size:
                return False

            z_score = abs(diff) / (np.sqrt(self.ewmv) + 1e-8)
            return z_score > self.threshold
        except Exception as e:
            print(f"Error in anomaly detection: {e}")
            return False

def create_data_sequence(length: int) -> List[float]:
    """
    Create a synthetic data sequence with anomalies.

    Args:
        length (int): The length of the data sequence to generate.

    Returns:
        List[float]: A list of float values representing the data sequence.
    """
    try:
        if not isinstance(length, int) or length <= 0:
            raise ValueError("Length must be a positive integer")

        t = np.linspace(0, 4*np.pi, length)
        base = 10 * np.sin(t) + 5 * np.cos(2*t)
        noise = np.random.normal(0, 1, length)
        trend = np.linspace(0, 5, length)
        data = base + noise + trend
        
        # Insert anomalies
        anomaly_positions = [100, 250, 400, 600, 750]
        for pos in anomaly_positions:
            if pos < length:
                data[pos] += np.random.uniform(10, 20)
        
        return data.tolist()
    except Exception as e:
        print(f"Error in data sequence creation: {e}")
        return []

def find_unusual_patterns(data: List[float]) -> List[Tuple[int, float]]:
    """
    Detect unusual patterns (anomalies) in the given data sequence.

    Args:
        data (List[float]): The input data sequence.

    Returns:
        List[Tuple[int, float]]: A list of tuples containing the index and value of detected anomalies.
    """
    detector = ModifiedAnomalyDetector()
    unusual_patterns = []
    
    for i, value in enumerate(data):
        try:
            if detector.update(value):
                unusual_patterns.append((i, value))
        except Exception as e:
            print(f"Error processing data point {i}: {e}")
    
    return unusual_patterns

def display_results(data: List[float], unusual_patterns: List[Tuple[int, float]]):
    """
    Display the results of anomaly detection.

    Args:
        data (List[float]): The input data sequence.
        unusual_patterns (List[Tuple[int, float]]): The detected anomalies.
    """
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(data, label='Data Sequence')
        if unusual_patterns:
            indices, values = zip(*unusual_patterns)
            plt.scatter(indices, values, color='red', label='Unusual Patterns')
        plt.title('Data Sequence with Detected Unusual Patterns')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()
    except Exception as e:
        print(f"Error in displaying results: {e}")

def main():
    """
    Main function to run the anomaly detection process.
    """
    try:
        # Generate data sequence
        data_sequence = create_data_sequence(1000)
        
        # Detect unusual patterns
        unusual_patterns = find_unusual_patterns(data_sequence)
        
        # Display results
        display_results(data_sequence, unusual_patterns)
        
        print(f"Detected {len(unusual_patterns)} unusual patterns.")
    except Exception as e:
        print(f"An error occurred in the main function: {e}")

if __name__ == "__main__":
    main()
