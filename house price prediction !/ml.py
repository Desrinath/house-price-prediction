import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Step 1: Import necessary libraries

# Step 2: Load the dataset
try:
    data = pd.read_csv('Housing.csv')
except FileNotFoundError:
    print("Error: Dataset 'Housing.csv' not found. Please provide the correct file path.")

# Step 3: Prepare the data
X = data[['area', 'bedrooms', 'bathrooms']]  # Features
y = data['price']  # Target variable

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2) Score: {r2}")

# Step 8: Save the trained model
joblib.dump(model, 'trained_model.pkl')
