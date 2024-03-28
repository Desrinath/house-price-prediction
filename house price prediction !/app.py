from flask import Flask, render_template, request, jsonify
import ml  # Import your trained model from ml.py
import numpy as np

app = Flask(__name__)

# Load the trained model
model = ml.model  # Access the trained model from ml.py

# Define the home route to render the index.html template
@app.route('/')
def index():
    return render_template('index1.html')

# Define a route to handle form submission and display the result
@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    area = float(request.form['area'])
    bedrooms = float(request.form['bedrooms'])
    bathrooms = float(request.form['bathrooms'])

    # Make prediction
    input_features = np.array([[area, bedrooms, bathrooms]])
    predicted_price = model.predict(input_features)

    # Render the result template with the predicted price
    return render_template('result.html', predicted_price=predicted_price[0])

if __name__ == '__main__':
    app.run(debug=True)
