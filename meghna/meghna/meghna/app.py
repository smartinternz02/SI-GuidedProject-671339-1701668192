from flask import Flask, render_template, request
import pickle
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
model = joblib.load('Lasso_regression_model.joblib')

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict')
def predict():
    return render_template("predict.html")

@app.route('/result', methods=["GET", "POST"])  # route to show the predictions in a web UI
def result():
    # reading the inputs given by the user
    input_feature = [float(x) for x in request.form.values() if x.strip()]  # Filter out empty strings
    if not input_feature:
        return "Please provide valid input values"
    
    x = [np.array(input_feature)]
    names = ["Milk and products", "Prepared meals, snacks, sweets etc.", "Health", "Personal care and effects", "Miscellaneous","Meat and fish","Food and beverages"]
    data = pd.DataFrame(x, columns=names)
    
    try:
        pred = model.predict(data)  # Assuming model.predict returns a numerical prediction
        return render_template("result.html", pred=pred)  # Pass pred as a variable to the template
    except Exception as e:
        print("Error during prediction:", e)
        return "Error occurred during prediction"

if __name__ == "__main__":
    app.run(debug=True, port=2222)
