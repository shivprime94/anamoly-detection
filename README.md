The ModifiedAnomalyDetector class uses an Exponentially Weighted Moving Average (EWMA) approach for anomaly detection. This algorithm is effective because:

- It adapts to changing data patterns over time, making it suitable for detecting anomalies in non-stationary data streams.
- It gives more weight to recent observations, allowing it to quickly respond to new trends.
- The use of exponential weighting provides a good balance between sensitivity to new data and stability against noise.
- It's computationally efficient, making it suitable for real-time anomaly detection in data streams.


<h3>Robust Error Handling and Data Validation:</h3>

- The code now includes try-except blocks to catch and handle potential errors.
- Input validation is performed in the ModifiedAnomalyDetector initialization and update method.
- The create_data_sequence function checks for valid input parameters.
- Error messages are printed to inform the user of any issues that occur during execution.

<h3>Libraries used:</h3>

- NumPy
- Matplotlib

Results:

<img width="1140" alt="Screenshot 2024-10-15 at 7 01 21 PM" src="https://github.com/user-attachments/assets/fe22d90d-5b2c-4577-8bae-c4537acb90d4">


Python Version: 3.x
