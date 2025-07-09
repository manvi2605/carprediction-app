from flask import Flask, render_template, request
import pandas as pd
import pickle
import logging

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.ERROR)

# Load the trained Random Forest model
with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define the feature list used during training
model_features = [
    'milage', 'car_age', 'clean_title_label_enc', 'accident_label_enc',
    'fuel_type_Gasoline', 'fuel_type_Hybrid',
    'brand_Ford', 'brand_BMW', 'brand_Jaguar', 'brand_Pontiac',
    'transmission_10-Speed A/T', 'transmission_6-Speed M/T',
    'transmission_6-Speed A/T', 'transmission_Transmission w/Dual Shift Mode',
    'transmission_A/T',
    'ext_col_Black', 'ext_col_White', 'ext_col_Gray', 'ext_col_Blue', 'ext_col_Purple',
    'int_col_Black', 'int_col_Gray', 'int_col_Beige', 'int_col_Brown'
]

@app.route('/')
def home():
    return render_template('index.html', error=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse form inputs
        milage = float(request.form['milage'])
        car_age = int(request.form['car_age'])
        clean_title = int(request.form['clean_title'])  # 1 or 0
        accident_label_enc = int(request.form['accident_label_enc'])  # 1 or 0

        fuel_type = request.form['fuel_type']
        brand = request.form['brand']
        transmission = request.form['transmission']
        ext_col = request.form['ext_col']
        int_col = request.form['int_col']

        if milage < 0 or car_age < 0:
            raise ValueError("Milage and car age must be non-negative.")

        # Initialize input with defaults
        input_dict = {feature: 0 for feature in model_features}

        # Set numeric inputs
        input_dict['milage'] = milage
        input_dict['car_age'] = car_age
        input_dict['clean_title_label_enc'] = clean_title
        input_dict['accident_label_enc'] = accident_label_enc

        # Safely one-hot encode
        one_hot_keys = [
            f'fuel_type_{fuel_type}',
            f'brand_{brand}',
            f'transmission_{transmission}',
            f'ext_col_{ext_col}',
            f'int_col_{int_col}'
        ]
        for key in one_hot_keys:
            if key in input_dict:
                input_dict[key] = 1  # set only if it exists in training features

        # Create DataFrame with correct column order
        input_df = pd.DataFrame([input_dict])[model_features]

        # Predict
        prediction = model.predict(input_df)[0]
        prediction = round(prediction, 2)

        return render_template('result.html', prediction=prediction)

    except Exception as e:
        logging.error(f"Prediction Error: {e}")
        return render_template('index.html', error="Prediction failed. Please check your inputs.")

if __name__ == '__main__':
    app.run(debug=True)
