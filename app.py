from flask import Flask, render_template, request
import pandas as pd
import pickle
import logging

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.ERROR)

# Load the trained model
with open('car_pred.pkl', 'rb') as file:
    model = pickle.load(file)

# Dynamically extract feature names from the model
model_features = model.feature_names_in_

# Home route
@app.route('/')
def home():
    return render_template('index.html', error=None)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract inputs from form
        milage = float(request.form['milage'])
        car_age = int(request.form['car_age'])
        clean_title = int(request.form['clean_title'])  # 1 or 0
        accident_label_enc = int(request.form['accident_label_enc'])  # 0 or 1

        fuel_type = request.form['fuel_type']
        brand = request.form['brand']
        transmission = request.form['transmission']
        ext_col = request.form['ext_col']
        int_col = request.form['int_col']

        # Validate inputs
        if milage < 0 or car_age < 0:
            raise ValueError("Milage and car age must be non-negative.")

        # Initialize input dictionary
        input_dict = {
            'milage': milage,
            'car_age': car_age,
            'clean_title_label_enc': clean_title,
            'accident_label_enc': accident_label_enc,
        }

        # Add default values for all one-hot encoded features
        for feature in model_features:
            if feature not in input_dict:
                input_dict[feature] = 0

        # Set the correct one-hot encoded values
        input_dict[f'fuel_type_{fuel_type}'] = 1
        input_dict[f'brand_{brand}'] = 1
        input_dict[f'transmission_{transmission}'] = 1
        input_dict[f'ext_col_{ext_col}'] = 1
        input_dict[f'int_col_{int_col}'] = 1

        # Create input DataFrame with proper order
        input_df = pd.DataFrame([input_dict])[model_features]

        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction = round(prediction, 2)

        return render_template('result.html', prediction=prediction)

    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        return render_template('index.html', error="An error occurred while processing your request. Please check your inputs.")

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
