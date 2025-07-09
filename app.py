from flask import Flask, render_template, request
import pandas as pd
import logging
import xgboost as xgb

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.ERROR)

# Load the trained XGBRegressor model from JSON
model = xgb.XGBRegressor()
model.load_model("xgb_reg_model.json")

# Define feature list used during training (must match exact training order)
model_features = [
    'milage', 'car_age', 'clean_title_label_enc', 'accident_label_enc',

    # One-hot encoded fuel types
    'fuel_type_Gasoline', 'fuel_type_Hybrid',

    # One-hot encoded brands
    'brand_Ford', 'brand_BMW', 'brand_Jaguar', 'brand_Pontiac',

    # One-hot encoded transmissions
    'transmission_10-Speed A/T', 'transmission_6-Speed M/T',
    'transmission_6-Speed A/T', 'transmission_Transmission w/Dual Shift Mode', 'transmission_A/T',

    # One-hot encoded exterior colors
    'ext_col_Black', 'ext_col_White', 'ext_col_Gray', 'ext_col_Blue', 'ext_col_Purple',

    # One-hot encoded interior colors
    'int_col_Black', 'int_col_Gray', 'int_col_Beige', 'int_col_Brown'
]

# Home route
@app.route('/')
def home():
    return render_template('index.html', error=None)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse user input
        milage = float(request.form['milage'])
        car_age = int(request.form['car_age'])
        clean_title = int(request.form['clean_title'])
        accident_label_enc = int(request.form['accident_label_enc'])

        fuel_type = request.form['fuel_type']
        brand = request.form['brand']
        transmission = request.form['transmission']
        ext_col = request.form['ext_col']
        int_col = request.form['int_col']

        # Basic validation
        if milage < 0 or car_age < 0:
            raise ValueError("Mileage and car age must be non-negative.")

        # Prepare input dictionary with default 0s
        input_dict = {feature: 0 for feature in model_features}

        # Assign numerical features
        input_dict['milage'] = milage
        input_dict['car_age'] = car_age
        input_dict['clean_title_label_enc'] = clean_title
        input_dict['accident_label_enc'] = accident_label_enc

        # One-hot encode categorical features
        encoded_keys = [
            f'fuel_type_{fuel_type}', f'brand_{brand}', f'transmission_{transmission}',
            f'ext_col_{ext_col}', f'int_col_{int_col}'
        ]
        for key in encoded_keys:
            if key in input_dict:
                input_dict[key] = 1

        # Create a DataFrame with correct order
        input_df = pd.DataFrame([input_dict])[model_features]

        # Predict using the XGBRegressor
        prediction = model.predict(input_df.values)[0]
        prediction = round(float(prediction), 2)

        return render_template('result.html', prediction=prediction)

    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        return render_template('index.html', error="Prediction failed. Please verify your input.")

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
