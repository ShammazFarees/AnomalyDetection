# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Create a larger synthetic dataset with 10,000 samples
data = pd.DataFrame({
    'duration': np.random.randint(1, 100, size=10000),
    'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], size=10000),
    'service': np.random.choice(['http', 'ftp', 'smtp', 'dns'], size=10000),
    'src_bytes': np.random.randint(0, 5000, size=10000),
    'dst_bytes': np.random.randint(0, 5000, size=10000),
    'attack_type': np.random.choice(['normal', 'anomaly'], size=10000, p=[0.8, 0.2])
})

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Data Preprocessing
# Check for missing values
print("\nChecking for missing values:")
print(data.isnull().sum())

# Identify numeric and categorical columns
numeric_cols = ['duration', 'src_bytes', 'dst_bytes']
categorical_cols = ['protocol_type', 'service']

# Debug: Print identified numeric and categorical columns
print("\nNumeric columns:")
print(numeric_cols)
print("\nCategorical columns:")
print(categorical_cols)

# Fill missing values for numeric columns with mean
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Fill missing values for categorical columns with mode
for col in categorical_cols:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Debug: Display the DataFrame after handling missing values
print("\nDataFrame after filling missing values:")
print(data)

# Encode target labels as integers
label_encoder = LabelEncoder()
data['attack_type'] = label_encoder.fit_transform(data['attack_type'])

# Debug: Display encoded target labels
print("\nEncoded target labels:")
print(data['attack_type'].unique())

# Feature Selection (assuming 'attack_type' is the target column)
X = data.drop(['attack_type'], axis=1)  # Features
y = data['attack_type']  # Target

# Print the columns of X to debug
print("\nColumns in X (features):")
print(X.columns.tolist())
print("\nFirst few rows of X (features):")
print(X.head())

# Preprocessing for numeric and categorical data
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')  # Ignore unknown categories

# Create a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Create a pipeline that first transforms the data and then fits the model
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', IsolationForest(contamination=0.2, random_state=42))])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the columns of X_train to debug
print("\nColumns in X_train:")
print(X_train.columns.tolist())

# Print shapes of X_train and X_test for debugging
print(f"\nShape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")

# Fit the model
try:
    pipeline.fit(X_train)
except ValueError as e:
    print(f"\nException during model fitting: {e}")
    print("\nColumn names in X_train:")
    print(X_train.columns.tolist())
    print("\nData in X_train:")
    print(X_train.head())
    raise  # Re-raise the exception to stop execution if fitting fails

# Predict anomalies in the test set
try:
    y_pred = pipeline.predict(X_test)
except AttributeError as e:
    print(f"\nException during prediction: {e}")
    print("\nPipeline details:")
    print(pipeline)
    print("\nPreprocessor details:")
    print(preprocessor)
    raise  # Re-raise the exception to stop execution if prediction fails

# Convert predictions: -1 -> 1 (anomaly), 1 -> 0 (normal)
y_pred = np.where(y_pred == -1, 1, 0)

# Ensure y_test is encoded similarly to y_pred
label_encoder = LabelEncoder()
y_test = label_encoder.fit_transform(y_test)
y_pred = label_encoder.inverse_transform(y_pred)

# Evaluate the model
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Visualize Anomalies (2D example)
plt.figure(figsize=(10, 6))
plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_pred, cmap='coolwarm', label='Prediction')
plt.title('Anomaly Detection Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Prediction')
plt.legend()
plt.show()

# Save the trained model
joblib.dump(pipeline, 'network_anomaly_detection_model.pkl')

print("\nModel saved successfully as 'network_anomaly_detection_model.pkl'")
