from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("flood_model.pkl")

# Load encoders
district_encoder = joblib.load("district_encoder.pkl")

# Month names for the dropdown
month_names = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December"
]

# Mapping month names back to numbers
month_mapping = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12
}


@app.route("/")
def home():
    return render_template(
        "index.html",
        districts=district_encoder.classes_,
        months=month_names
    )


@app.route("/predict", methods=["POST"])
def predict():

    # Get form values
    district_name = request.form["district"]
    month_name = request.form["month"]
    rainfall = float(request.form["rainfall"])

    # Encode district
    district = district_encoder.transform([district_name])[0]

    # Convert month name to number
    month = month_mapping[month_name]

    # Make prediction
    prediction = model.predict([[district, month, rainfall]])

    if prediction[0] == 1:
        result = "HIGH FLOOD RISK"
        status = "high"

        advice = [
            "Avoid travelling to low-lying areas.",
            "Keep emergency contacts ready.",
            "Stay updated with official weather alerts.",
            "Move to safer locations if required."
        ]

    else:
        result = "LOW FLOOD RISK"
        status = "low"

        advice = [
            "No immediate flood danger.",
            "Continue monitoring weather forecasts.",
            "Stay alert during heavy rainfall.",
            "Follow local safety advisories."
        ]

    return render_template(
        "index.html",
        prediction=result,
        status=status,
        advice=advice,
        districts=district_encoder.classes_,
        months=month_names
    )


if __name__ == "__main__":
    app.run(debug=True)