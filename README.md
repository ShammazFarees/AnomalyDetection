# Anomaly Detection
# Network Anomaly Detection

This project aims to detect anomalies in network traffic using machine learning techniques. The Isolation Forest algorithm is employed to identify normal and anomalous network behaviors. The project includes data preprocessing, model training, evaluation, and visualization.

## Table of Contents

- [Dataset](#dataset)
- [Libraries and Tools](#libraries-and-tools)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Visualization](#visualization)
- [Saving the Model](#saving-the-model)
- [Usage](#usage)

## Dataset

A synthetic dataset with 10,000 samples is created for this project. The dataset includes the following features:
- `duration`: Duration of the connection
- `protocol_type`: Type of protocol (TCP, UDP, ICMP)
- `service`: Network service on the destination (e.g., HTTP, FTP)
- `src_bytes`: Number of data bytes from the source to the destination
- `dst_bytes`: Number of data bytes from the destination to the source
- `attack_type`: Type of network traffic (`normal` or `anomaly`)

## Libraries and Tools

The following libraries and tools are used in this project:
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib

## Data Preprocessing

1. **Handling Missing Values**:
   - Numeric columns: Fill missing values with the mean.
   - Categorical columns: Fill missing values with the mode.

2. **Encoding Categorical Variables**:
   - Encode categorical variables using `OneHotEncoder`.

3. **Label Encoding**:
   - Encode the target variable (`attack_type`) using `LabelEncoder`.

## Model Training

The Isolation Forest algorithm is used for anomaly detection. A pipeline is created to preprocess the data and fit the model.

1. **Train-Test Split**:
   - Split the data into training and testing sets (80% training, 20% testing).

2. **Pipeline Creation**:
   - Create a pipeline with a column transformer for preprocessing and the Isolation Forest model for training.

## Model Evaluation

Evaluate the model using classification report and confusion matrix.

## Visualization

Visualize anomalies in a 2D scatter plot for better understanding.

## Saving the Model

The trained model is saved as `network_anomaly_detection_model.pkl` using Joblib.

## Usage

To use the trained model, load it using Joblib and predict anomalies on new data.

```python
import joblib

# Load the trained model
pipeline = joblib.load('network_anomaly_detection_model.pkl')

# Predict anomalies on new data
new_data = pd.DataFrame({
    'duration': [50],
    'protocol_type': ['tcp'],
    'service': ['http'],
    'src_bytes': [1000],
    'dst_bytes': [500]
})

prediction = pipeline.predict(new_data)
print(f"Anomaly Prediction: {prediction}")
Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

License
This project is licensed under the MIT License.

