import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from typing import List, Tuple

class ModifiedAnomalyDetector:
    def __init__(self, window_size: int = 50, threshold: float = 2.5):
        self.window_size = window_size
        self.threshold = threshold
        self.window = deque(maxlen=window_size)
        self.ewma = None
        self.ewmv = None
        self.alpha = 0.1

    def update(self, value: float) -> bool:
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

def create_data_sequence(length: int) -> List[float]:
    t = np.linspace(0, 4*np.pi, length)
    base = 10 * np.sin(t) + 5 * np.cos(2*t)
    noise = np.random.normal(0, 1, length)
    trend = np.linspace(0, 5, length)
    data = base + noise + trend
    
    # Insert anomalies
    anomaly_positions = [100, 250, 400, 600, 750]
    for pos in anomaly_positions:
        data[pos] += np.random.uniform(10, 20)
    
    return data.tolist()

def find_unusual_patterns(data: List[float]) -> List[Tuple[int, float]]:
    detector = ModifiedAnomalyDetector()
    unusual_patterns = []
    
    for i, value in enumerate(data):
        if detector.update(value):
            unusual_patterns.append((i, value))
    
    return unusual_patterns

def display_results(data: List[float], unusual_patterns: List[Tuple[int, float]]):
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

def main():
    # Generate data sequence
    data_sequence = create_data_sequence(1000)
    
    # Detect unusual patterns
    unusual_patterns = find_unusual_patterns(data_sequence)
    
    # Display results
    display_results(data_sequence, unusual_patterns)
    
    print(f"Detected {len(unusual_patterns)} unusual patterns.")

if __name__ == "__main__":
    main()
