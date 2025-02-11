"""
Greek Text Corruption Classifier

This script applies a pre-trained machine learning model to classify corruption in Greek text documents
from the Pergamos dataset. It processes a CSV input file and generates predictions for each text entry.

Dependencies:
    - pickle: For loading the trained model
    - pandas: For data manipulation
    - time: For performance monitoring
"""

import pickle
import pandas
import time

# Track execution time for performance monitoring
start_time = time.time()

# Load the pre-trained corruption classification model from disk
model = pickle.load(open('corruption_classification_model.pkl', 'rb'))

# Load and prepare input data from CSV
# Note: Using a sample of 100 documents from Pergamos dataset
data = pandas.read_csv('data/pergamos_100.csv')

# Remove any rows with missing text values to ensure clean processing
data.dropna(subset=['text'], inplace=True)

# Apply the model to generate corruption predictions
predictions = model.predict(data) 

# Add predictions to the column 'value' in the dataset
data['value'] = predictions

# Display sample of results for verification
print(data.head())

# Save the classified dataset to a new CSV file
# Note: Output filename matches input with '_predicted' suffix
data.to_csv('data/pergamos_100_predicted.csv', index=False)

# Calculate and display total processing time
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.4f} seconds")
