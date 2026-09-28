from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# ============================================================
# PATHS
# ============================================================

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(APP_DIR)

DATASET_FILE = os.path.join(
    PROJECT_DIR,
    "dataset",
    "raw_data",
    "TN_IMD_Risk_Dataset.csv"
)


# ============================================================
# LOAD MACHINE LEARNING MODELS
# ============================================================

random_forest_model = joblib.load(
    os.path.join(APP_DIR, "random_forest_model.pkl")
)

decision_tree_model = joblib.load(
    os.path.join(APP_DIR, "decision_tree_model.pkl")
)

logistic_regression_model = joblib.load(
    os.path.join(APP_DIR, "logistic_regression_model.pkl")
)

scaler = joblib.load(
    os.path.join(APP_DIR, "scaler.pkl")
)

district_encoder = joblib.load(
    os.path.join(APP_DIR, "district_encoder.pkl")
)

risk_encoder = joblib.load(
    os.path.join(APP_DIR, "risk_encoder.pkl")
)


# ============================================================
# MONTH INFORMATION
# ============================================================

MONTH_NUMBERS = {
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


MONTH_ABBREVIATIONS = {
    "January": "Jan",
    "February": "Feb",
    "March": "Mar",
    "April": "Apr",
    "May": "May",
    "June": "Jun",
    "July": "Jul",
    "August": "Aug",
    "September": "Sep",
    "October": "Oct",
    "November": "Nov",
    "December": "Dec"
}


# ============================================================
# NORMALIZE MONTH
# ============================================================

def normalize_month(month_value):

    month_map = {
        "1": "January",
        "2": "February",
        "3": "March",
        "4": "April",
        "5": "May",
        "6": "June",
        "7": "July",
        "8": "August",
        "9": "September",
        "10": "October",
        "11": "November",
        "12": "December"
    }

    month_value = str(month_value).strip()

    if month_value in month_map:
        return month_map[month_value]

    if month_value in MONTH_NUMBERS:
        return month_value

    return None


# ============================================================
# LOAD RAINFALL DATASET
# ============================================================

def load_rainfall_dataset():

    if not os.path.exists(DATASET_FILE):

        print("Rainfall dataset not found:")
        print(DATASET_FILE)

        return None

    try:

        df = pd.read_csv(DATASET_FILE)

        return df

    except Exception as error:

        print("Dataset loading error:", error)

        return None


# ============================================================
# GET RAINFALL FROM DATASET
# ============================================================

def get_rainfall_from_dataset(
    district_name,
    month_name
):

    df = load_rainfall_dataset()

    if df is None:
        return None

    try:

        normalized_month = normalize_month(
            month_name
        )

        if normalized_month is None:
            return None

        dataset_month = MONTH_ABBREVIATIONS.get(
            normalized_month
        )

        if dataset_month is None:
            return None

        filtered = df[
            (
                df["District"].str.upper()
                ==
                district_name.upper()
            )
            &
            (
                df["Month"].str.upper()
                ==
                dataset_month.upper()
            )
        ]

        if filtered.empty:
            return None

        rainfall = filtered["Rainfall_mm"].mean()

        return round(float(rainfall), 2)

    except Exception as error:

        print(
            "Rainfall calculation error:",
            error
        )

        return None


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html",

        districts=district_encoder.classes_,

        months=list(
            MONTH_NUMBERS.keys()
        ),

        prediction=None,

        status=None,

        advice=[],

        rf_prediction=None,

        dt_prediction=None,

        lr_prediction=None
    )


# ============================================================
# AUTOMATIC RAINFALL ROUTE
# ============================================================

@app.route("/get_rainfall")
def get_rainfall():

    district_name = request.args.get(
        "district",
        ""
    )

    month_value = request.args.get(
        "month",
        ""
    )

    if not district_name or not month_value:

        return jsonify({
            "rainfall": None
        })

    rainfall = get_rainfall_from_dataset(
        district_name,
        month_value
    )

    return jsonify({
        "rainfall": rainfall
    })


# ============================================================
# FLOOD PREDICTION
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    district_name = request.form.get(
        "district",
        ""
    )

    month_value = request.form.get(
        "month",
        ""
    )

    rainfall_text = request.form.get(
        "rainfall",
        ""
    )


    # --------------------------------------------------------
    # CHECK DISTRICT AND MONTH
    # --------------------------------------------------------

    if not district_name or not month_value:

        return render_template(
            "index.html",

            districts=district_encoder.classes_,

            months=list(
                MONTH_NUMBERS.keys()
            ),

            prediction="Please select a district and month.",

            status="low",

            advice=[],

            rf_prediction=None,

            dt_prediction=None,

            lr_prediction=None
        )


    # --------------------------------------------------------
    # CONVERT MONTH TO FULL NAME
    # --------------------------------------------------------

    month_name = normalize_month(
        month_value
    )

    if month_name is None:

        return render_template(
            "index.html",

            districts=district_encoder.classes_,

            months=list(
                MONTH_NUMBERS.keys()
            ),

            prediction="Invalid month selected.",

            status="low",

            advice=[],

            rf_prediction=None,

            dt_prediction=None,

            lr_prediction=None
        )


    # --------------------------------------------------------
    # GET RAINFALL
    # --------------------------------------------------------

    try:

        rainfall = float(
            rainfall_text
        )

    except (
        ValueError,
        TypeError
    ):

        rainfall = get_rainfall_from_dataset(
            district_name,
            month_name
        )

        if rainfall is None:

            return render_template(
                "index.html",

                districts=district_encoder.classes_,

                months=list(
                    MONTH_NUMBERS.keys()
                ),

                prediction="Rainfall data is not available.",

                status="low",

                advice=[],

                rf_prediction=None,

                dt_prediction=None,

                lr_prediction=None
            )


    # --------------------------------------------------------
    # ENCODE DISTRICT
    # --------------------------------------------------------

    try:

        district_code = (
            district_encoder.transform(
                [district_name]
            )[0]
        )

    except Exception:

        return render_template(
            "index.html",

            districts=district_encoder.classes_,

            months=list(
                MONTH_NUMBERS.keys()
            ),

            prediction="Selected district is not available.",

            status="low",

            advice=[],

            rf_prediction=None,

            dt_prediction=None,

            lr_prediction=None
        )


    # --------------------------------------------------------
    # CONVERT MONTH NAME TO NUMBER INTERNALLY
    # --------------------------------------------------------

    month_number = MONTH_NUMBERS.get(
        month_name
    )

    if month_number is None:

        return render_template(
            "index.html",

            districts=district_encoder.classes_,

            months=list(
                MONTH_NUMBERS.keys()
            ),

            prediction="Invalid month selected.",

            status="low",

            advice=[],

            rf_prediction=None,

            dt_prediction=None,

            lr_prediction=None
        )


    # --------------------------------------------------------
    # REFERENCE YEAR
    # --------------------------------------------------------

    year = 2010


    # --------------------------------------------------------
    # MODEL INPUT
    # --------------------------------------------------------

    input_data = [[
        district_code,
        year,
        month_number,
        rainfall
    ]]


    # ========================================================
    # RANDOM FOREST PREDICTION
    # ========================================================

    rf_prediction_code = (
        random_forest_model.predict(
            input_data
        )[0]
    )

    rf_result = (
        risk_encoder.inverse_transform(
            [rf_prediction_code]
        )[0]
    )


    # ========================================================
    # DECISION TREE PREDICTION
    # ========================================================

    dt_prediction_code = (
        decision_tree_model.predict(
            input_data
        )[0]
    )

    dt_result = (
        risk_encoder.inverse_transform(
            [dt_prediction_code]
        )[0]
    )


    # ========================================================
    # LOGISTIC REGRESSION PREDICTION
    # ========================================================

    scaled_input = scaler.transform(
        input_data
    )

    lr_prediction_code = (
        logistic_regression_model.predict(
            scaled_input
        )[0]
    )

    lr_result = (
        risk_encoder.inverse_transform(
            [lr_prediction_code]
        )[0]
    )


    # ========================================================
    # COMBINE ALL THREE MODELS
    # ========================================================

    predictions = [
        rf_result,
        dt_result,
        lr_result
    ]


    high_count = predictions.count(
        "High"
    )

    low_count = predictions.count(
        "Low"
    )


    # --------------------------------------------------------
    # MAJORITY VOTING
    # --------------------------------------------------------

    if high_count >= 2:

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


    # ========================================================
    # SHOW RESULT
    # ========================================================

    return render_template(
        "index.html",

        prediction=result,

        status=status,

        advice=advice,

        rf_prediction=rf_result,

        dt_prediction=dt_result,

        lr_prediction=lr_result,

        districts=district_encoder.classes_,

        months=list(
            MONTH_NUMBERS.keys()
        )
    )


# ============================================================
# PAGE 2 — HISTORICAL ANALYSIS
# ============================================================

@app.route("/analysis")
def analysis():

    return render_template(
        "analysis.html",
        districts=district_encoder.classes_
    )


# ============================================================
# PAGE 2 — HISTORICAL ANNUAL DATA
# ============================================================

@app.route("/historical_data")
def historical_data():

    district_name = request.args.get(
        "district",
        ""
    )

    if not district_name:

        return jsonify({
            "success": False
        })


    try:

        df = pd.read_csv(
            os.path.join(
                PROJECT_DIR,
                "dataset",
                "raw_data",
                "TN_IMD_District_Rainfall_1901_2010.csv"
            )
        )


        filtered = df[
            df["District"].str.upper()
            ==
            district_name.upper()
        ]


        if filtered.empty:

            return jsonify({
                "success": False
            })


        filtered = filtered.dropna(
            subset=["Annual"]
        )


        filtered = filtered.sort_values(
            by="Year"
        )


        years = (
            filtered["Year"]
            .astype(int)
            .tolist()
        )


        rainfall_values = (
            filtered["Annual"]
            .astype(float)
            .round(2)
            .tolist()
        )


        average_rainfall = round(
            filtered["Annual"].mean(),
            2
        )


        maximum_rainfall = round(
            filtered["Annual"].max(),
            2
        )


        minimum_rainfall = round(
            filtered["Annual"].min(),
            2
        )


        return jsonify({

            "success": True,

            "district": district_name,

            "years": years,

            "rainfall": rainfall_values,

            "average": average_rainfall,

            "maximum": maximum_rainfall,

            "minimum": minimum_rainfall

        })


    except Exception as error:

        print(
            "Historical data error:",
            error
        )


        return jsonify({
            "success": False
        })


# ============================================================
# PAGE 2 — MONTHLY RAINFALL PATTERN
# ============================================================

@app.route("/monthly_rainfall")
def monthly_rainfall():

    district_name = request.args.get(
        "district",
        ""
    )

    if not district_name:

        return jsonify({
            "success": False
        })


    try:

        df = pd.read_csv(
            os.path.join(
                PROJECT_DIR,
                "dataset",
                "raw_data",
                "TN_IMD_District_Rainfall_1901_2010.csv"
            )
        )


        filtered = df[
            df["District"].str.upper()
            ==
            district_name.upper()
        ]


        if filtered.empty:

            return jsonify({
                "success": False
            })


        # ====================================================
        # MONTH NAMES
        # ====================================================

        months = [

            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec"

        ]


        monthly_rainfall = []


        # ====================================================
        # CALCULATE AVERAGE RAINFALL FOR EACH MONTH
        # ====================================================

        for month in months:

            values = pd.to_numeric(
                filtered[month],
                errors="coerce"
            ).dropna()


            if values.empty:

                monthly_rainfall.append(
                    0
                )

            else:

                monthly_rainfall.append(
                    round(
                        values.mean(),
                        2
                    )
                )


        return jsonify({

            "success": True,

            "district": district_name,

            "months": months,

            "rainfall": monthly_rainfall

        })


    except Exception as error:

        print(
            "Monthly rainfall error:",
            error
        )


        return jsonify({
            "success": False
        })


# ============================================================
# PAGE 3 — MODEL PERFORMANCE
# ============================================================

@app.route("/model-performance")
def model_performance():

    metrics_path = os.path.join(
        PROJECT_DIR,
        "dataset",
        "raw_data",
        "model_metrics.csv"
    )

    metrics = pd.read_csv(
        metrics_path
    )

    return render_template(
        "model_performance.html",
        metrics=metrics.to_dict(
            orient="records"
        )
    )


# ============================================================
# RUN FLASK APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )