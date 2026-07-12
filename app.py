from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load Model
model = joblib.load("model.pkl")
pipeline = joblib.load("pipline.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    longitude = float(request.form["longitude"])
    latitude = float(request.form["latitude"])
    housing_median_age = float(request.form["housing_median_age"])
    total_rooms = float(request.form["total_rooms"])
    total_bedrooms = float(request.form["total_bedrooms"])
    population = float(request.form["population"])
    households = float(request.form["households"])
    median_income = float(request.form["median_income"])
    ocean_proximity = request.form["ocean_proximity"]

    data = pd.DataFrame([{
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity
    }])

    transformed = pipeline.transform(data)

    prediction = model.predict(transformed)[0]

    prediction = round(prediction, 2)

    return render_template(
        "index.html",
        prediction_text=f"Estimated House Price : ${prediction:,.2f}"
    )


if __name__ == "__main__":
    app.run(debug=True)