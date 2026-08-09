from flask import Flask, render_template, request, jsonify
import joblib
import csv
import os
import re

app = Flask(__name__)

# ---------------------------------------------------
# LOAD TRAINED MODEL
# ---------------------------------------------------

model = joblib.load("flood_model.pkl")

# ---------------------------------------------------
# LOAD ENCODERS
# ---------------------------------------------------

district_encoder = joblib.load("district_encoder.pkl")
month_encoder = joblib.load("month_encoder.pkl")


# ---------------------------------------------------
# MONTH NAME NORMALIZATION
# ---------------------------------------------------

def normalize_month(month):
    month = str(month).strip().lower()

    months = {
        "1": "january",
        "jan": "january",
        "january": "january",

        "2": "february",
        "feb": "february",
        "february": "february",

        "3": "march",
        "mar": "march",
        "march": "march",

        "4": "april",
        "apr": "april",
        "april": "april",

        "5": "may",
        "may": "may",

        "6": "june",
        "jun": "june",
        "june": "june",

        "7": "july",
        "jul": "july",
        "july": "july",

        "8": "august",
        "aug": "august",
        "august": "august",

        "9": "september",
        "sep": "september",
        "sept": "september",
        "september": "september",

        "10": "october",
        "oct": "october",
        "october": "october",

        "11": "november",
        "nov": "november",
        "november": "november",

        "12": "december",
        "dec": "december",
        "december": "december"
    }

    return months.get(month, month)


# ---------------------------------------------------
# FIND RAINFALL FROM DATASET
# ---------------------------------------------------

def get_rainfall_from_dataset(district_name, month_name):

    # Dataset location
    dataset_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "dataset",
        "raw_data",
        "TamilNadu_Rainfall.csv"
    )

    # If dataset is not found, try the current folder
    if not os.path.exists(dataset_path):

        dataset_path = os.path.join(
            os.path.dirname(__file__),
            "TamilNadu_Rainfall.csv"
        )

    if not os.path.exists(dataset_path):
        return None

    district_name = district_name.strip().lower()
    month_name = normalize_month(month_name)

    rainfall_values = []

    try:

        with open(
            dataset_path,
            "r",
            encoding="utf-8-sig"
        ) as file:

            reader = csv.DictReader(file)

            if not reader.fieldnames:
                return None

            columns = reader.fieldnames

            # Find district column
            district_column = None

            for column in columns:

                clean_column = column.strip().lower()

                if clean_column in [
                    "district",
                    "district_name",
                    "district name"
                ]:

                    district_column = column
                    break

            # Find month column
            month_column = None

            for column in columns:

                clean_column = column.strip().lower()

                if clean_column in [
                    "month",
                    "months"
                ]:

                    month_column = column
                    break

            if district_column is None:
                return None

            # ---------------------------------------------------
            # READ EACH ROW
            # ---------------------------------------------------

            for row in reader:

                row_district = str(
                    row.get(district_column, "")
                ).strip().lower()

                if row_district != district_name:
                    continue

                # If dataset has a month column
                if month_column is not None:

                    row_month = normalize_month(
                        row.get(month_column, "")
                    )

                    if row_month != month_name:
                        continue

                # ---------------------------------------------------
                # FIRST: LOOK FOR TOTAL RAINFALL COLUMN
                # ---------------------------------------------------

                total_found = False

                for column in columns:

                    clean_column = (
                        column.strip()
                        .lower()
                        .replace(" ", "")
                        .replace("_", "")
                    )

                    if (
                        "total" in clean_column
                        and "rainfall" in clean_column
                    ):

                        value = row.get(column, "")

                        try:

                            value = float(
                                str(value)
                                .replace(",", "")
                                .strip()
                            )

                            rainfall_values.append(value)
                            total_found = True

                        except (ValueError, TypeError):
                            pass

                        break

                if total_found:
                    continue

                # ---------------------------------------------------
                # OTHERWISE ADD DAILY RAINFALL COLUMNS
                # ---------------------------------------------------

                daily_total = 0.0
                found_daily_value = False

                for column in columns:

                    clean_column = column.strip().lower()

                    # Match columns such as:
                    # 1, 2, 3
                    # day1, day2
                    # day_1
                    # 1st, 2nd, 3rd

                    if (
                        clean_column.isdigit()
                        or clean_column.startswith("day")
                        or re.match(
                            r"^\d+(st|nd|rd|th)$",
                            clean_column
                        )
                    ):

                        try:

                            value = float(
                                str(row.get(column, ""))
                                .replace(",", "")
                                .strip()
                            )

                            daily_total += value
                            found_daily_value = True

                        except (ValueError, TypeError):
                            pass

                if found_daily_value:
                    rainfall_values.append(daily_total)

        # ---------------------------------------------------
        # CALCULATE AVERAGE
        # ---------------------------------------------------

        if rainfall_values:

            average_rainfall = (
                sum(rainfall_values)
                / len(rainfall_values)
            )

            return round(average_rainfall, 2)

    except Exception as error:

        print(
            "Rainfall dataset error:",
            error
        )

    return None


# ---------------------------------------------------
# HOME PAGE
# ---------------------------------------------------

@app.route("/")
def home():

    return render_template(
        "index.html",
        districts=district_encoder.classes_,
        months=month_encoder.classes_
    )


# ---------------------------------------------------
# AUTOMATIC RAINFALL ROUTE
# ---------------------------------------------------

@app.route("/get_rainfall")
def get_rainfall():

    district_name = request.args.get(
        "district",
        ""
    )

    month_name = request.args.get(
        "month",
        ""
    )

    # Keep rainfall blank until both are selected
    if not district_name or not month_name:

        return jsonify({
            "rainfall": None
        })

    rainfall = get_rainfall_from_dataset(
        district_name,
        month_name
    )

    return jsonify({
        "rainfall": rainfall
    })


# ---------------------------------------------------
# FLOOD PREDICTION
# ---------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    # Get form values

    district_name = request.form["district"]

    month_name = request.form["month"]

    rainfall = float(
        request.form["rainfall"]
    )

    # Convert names into encoded values

    district = district_encoder.transform(
        [district_name]
    )[0]

    month = month_encoder.transform(
        [month_name]
    )[0]

    # Predict

    prediction = model.predict(
        [[district, month, rainfall]]
    )

    # ---------------------------------------------------
    # HIGH RISK
    # ---------------------------------------------------

    if prediction[0] == 1:

        result = "HIGH FLOOD RISK"

        status = "high"

        advice = [

            "Avoid travelling to low-lying areas.",

            "Keep emergency contacts ready.",

            "Stay updated with official weather alerts.",

            "Move to safer locations if required."

        ]

    # ---------------------------------------------------
    # LOW RISK
    # ---------------------------------------------------

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

        months=month_encoder.classes_

    )


# ---------------------------------------------------
# RUN APPLICATION
# ---------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )