from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model, encoder, and scaler
try:
    model = joblib.load('./models/model.pkl')
    encoder = joblib.load('./models/encoder.pkl')
    scaler = joblib.load('./models/scaler.pkl')
except Exception as e:
    print(f"Error loading model: {e}")

# Define the numeric and categorical columns
numeric_cols = ['Sunlight_Hours', 'Temperature', 'Humidity']
categorical_cols = ['Soil_Type', 'Water_Frequency', 'Fertilizer_Type']


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the form
        data = request.get_json()
        df = pd.DataFrame([data])
        print(data)
        print(df)

        # Handle categorical columns
        df_encoded = pd.DataFrame(encoder.transform(df[categorical_cols]), columns=encoder.get_feature_names_out(categorical_cols))

        # Drop original categorical columns and add encoded ones
        df = df.drop(categorical_cols, axis=1).join(df_encoded)

        # Scale numeric columns
        df[numeric_cols] = scaler.transform(df[numeric_cols])
        print("final df before prediction")
        print(df)

        # Predict using the model
        prediction = model.predict(df)[0]
        print("prediction value: ", prediction)

        return jsonify({'prediction': int(prediction)})
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'prediction': "An error occurred during prediction"})


if __name__ == '__main__':
    app.run(debug=True)
