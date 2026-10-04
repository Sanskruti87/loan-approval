from flask import Flask, render_template, request
import pandas as pd
import joblib


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("model/loan_model.pkl")


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ----------------------------------------------------
        # GET DATA FROM HTML FORM
        # ----------------------------------------------------

        applicant_id = request.form["applicant_id"]

        gender = request.form["gender"]

        married = request.form["married"]

        dependents = request.form["dependents"]

        education = request.form["education"]

        employment = request.form["employment"]

        applicant_income = float(
            request.form["applicant_income"]
        )

        coapplicant_income = float(
            request.form["coapplicant_income"]
        )

        loan_amount = float(
            request.form["loan_amount"]
        )

        loan_term = int(
            request.form["loan_term"]
        )

        credit_history = float(
            request.form["credit_history"]
        )

        property_area = request.form["property_area"]


        # ----------------------------------------------------
        # EMPLOYMENT CONVERSION
        # ----------------------------------------------------

        if employment == "Self-employed":

            self_employed = "Yes"

        else:

            self_employed = "No"


        # ----------------------------------------------------
        # LOAN AMOUNT
        # ----------------------------------------------------

        loan_amount_for_model = (
            loan_amount / 1000
        )


        # ----------------------------------------------------
        # CREATE DATAFRAME
        # ----------------------------------------------------

        input_data = pd.DataFrame({

            "Married": [
                married
            ],

            "Dependents": [
                dependents
            ],

            "Education": [
                education
            ],

            "Self_Employed": [
                self_employed
            ],

            "Property_Area": [
                property_area
            ],

            "ApplicantIncome": [
                applicant_income
            ],

            "CoapplicantIncome": [
                coapplicant_income
            ],

            "LoanAmount": [
                loan_amount_for_model
            ],

            "Loan_Amount_Term": [
                loan_term
            ],

            "Credit_History": [
                credit_history
            ]

        })


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        prediction_value = model.predict(
            input_data
        )[0]


        probabilities = model.predict_proba(
            input_data
        )[0]


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        if prediction_value == 1:

            prediction = "Approved"

        else:

            prediction = "Rejected"


        approval_probability = round(
            probabilities[1] * 100,
            2
        )


        rejection_probability = round(
            probabilities[0] * 100,
            2
        )


        # ----------------------------------------------------
        # SEND RESULT TO HTML
        # ----------------------------------------------------

        return render_template(

            "index.html",

            prediction=prediction,

            approval_probability=approval_probability,

            rejection_probability=rejection_probability

        )


    except Exception as e:

        return f"""
        <h2>Prediction Error</h2>
        <p>{str(e)}</p>
        """


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )