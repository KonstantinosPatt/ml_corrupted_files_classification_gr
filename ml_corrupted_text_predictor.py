import pickle
import pandas
import time

start_time = time.time()

# Load the model and preprocessor back
model = pickle.load(open('corruption_classification_model.pkl', 'rb'))

# Load a text
data = pandas.read_csv('data/pergamos_100.csv')
data.dropna(subset=['text'], inplace=True)

# Predict the class of the new data
predictions = model.predict(data) 

data['value'] = predictions

print(data.head())

data.to_csv('data/pergamos_100_predicted.csv', index=False)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.4f} seconds")
