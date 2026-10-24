from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load("model/loan_model.pkl")

# General document checklist examples for India.
# Exact requirements vary by bank, applicant profile, and loan scheme.
DOCUMENTS_BY_LOAN_TYPE = {
    "Personal Loan": [
        "PAN card and Aadhaar card / other accepted identity proof",
        "Address proof (if not covered by identity proof)",
        "Recent passport-size photographs, if requested by the bank",
        "Last 3–6 months' bank statements",
        "Salary slips for the last 3–6 months (salaried applicants)",
        "Employment proof / employee ID or appointment letter, if requested",
        "For self-employed applicants: ITRs, business proof, and financial statements as requested"
    ],
    "Home Loan": [
        "PAN card and Aadhaar card / other accepted identity proof",
        "Address proof and passport-size photographs, if requested",
        "Salary slips and employment proof (salaried applicants) OR ITRs and business financials (self-employed applicants)",
        "Last 6 months' bank statements",
        "Property documents and title/ownership documents",
        "Agreement for sale / allotment letter and payment receipts, where applicable",
        "Approved building plan and relevant property approvals, where applicable",
        "Latest property tax receipts / encumbrance-related documents, if requested"
    ],
    "Education Loan": [
        "PAN card and Aadhaar card / other accepted identity proof of applicant and co-applicant",
        "Address proof and photographs, if requested",
        "Admission letter from the recognised institution",
        "Course details and fee structure / estimate of expenses",
        "Previous academic mark sheets and certificates",
        "Co-applicant's income proof and recent bank statements",
        "Scholarship letter, if applicable",
        "Collateral/property documents only if required for the selected loan scheme"
    ],
    "Vehicle Loan": [
        "PAN card and Aadhaar card / other accepted identity proof",
        "Address proof and photographs, if requested",
        "Recent salary slips or self-employment income proof",
        "Last 3–6 months' bank statements",
        "Vehicle quotation / pro forma invoice from the dealer",
        "Driving licence, if requested by the lender",
        "Co-applicant or guarantor documents, if applicable"
    ],
    "Business Loan": [
        "PAN card and Aadhaar card / other accepted identity proof of proprietor/partners/directors",
        "Business address proof and business registration / constitution documents",
        "Last 6–12 months' business bank statements",
        "Income Tax Returns and financial statements for the period requested by the bank",
        "GST returns / GST registration, if applicable",
        "Business plan, cash-flow estimates, or project report, if requested",
        "Details of existing loans and repayment obligations",
        "Collateral documents and guarantor details, if applicable"
    ],
    "Agriculture Loan": [
        "PAN card and Aadhaar card / other accepted identity proof",
        "Address proof and recent photographs, if requested",
        "Land ownership / land-record documents or valid tenancy/lease documents, as applicable",
        "Details of crops, cultivation, or proposed agricultural activity",
        "Bank account details and recent bank statements",
        "Estimate/quotation for seeds, fertilisers, equipment, or other intended use, if requested",
        "Kisan Credit Card or previous agricultural loan details, if applicable"
    ],
    "Gold Loan": [
        "PAN card and Aadhaar card / other accepted identity proof",
        "Address proof, if required",
        "Gold ornaments/items to be pledged for lender appraisal",
        "Bank account details",
        "Any declaration or supporting documents required by the lender"
    ]
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        applicant_id = request.form["applicant_id"]
        loan_type = request.form["loan_type"]
        married = request.form["married"]
        dependents = request.form["dependents"]
        education = request.form["education"]
        employment = request.form["employment"]
        applicant_income = float(request.form["applicant_income"])
        coapplicant_income = float(request.form.get("coapplicant_income") or 0)
        loan_amount = float(request.form["loan_amount"])
        loan_term = int(request.form["loan_term"])
        credit_history = float(request.form["credit_history"])
        property_area = request.form["property_area"]

        self_employed = "Yes" if employment == "Self-employed" else "No"

        # This dataset/model was trained without loan type. Keep its expected feature
        # schema unchanged; loan_type is currently used for the document checklist only.
        # Dataset LoanAmount is generally in thousands, so convert rupees to thousands.
        input_data = pd.DataFrame({
            "Married": [married],
            "Dependents": [dependents],
            "Education": [education],
            "Self_Employed": [self_employed],
            "Property_Area": [property_area],
            "ApplicantIncome": [applicant_income],
            "CoapplicantIncome": [coapplicant_income],
            "LoanAmount": [loan_amount / 1000],
            "Loan_Amount_Term": [loan_term],
            "Credit_History": [credit_history]
        })

        prediction_value = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        prediction = "Approved" if prediction_value == 1 else "Rejected"

        # Find probability values by class label rather than assuming class order.
        class_probabilities = dict(zip(model.classes_, probabilities))
        approval_probability = round(float(class_probabilities.get(1, 0)) * 100, 2)
        rejection_probability = round(float(class_probabilities.get(0, 0)) * 100, 2)

        required_documents = DOCUMENTS_BY_LOAN_TYPE.get(loan_type, []) if prediction == "Approved" else []

        return render_template(
            "index.html",
            prediction=prediction,
            approval_probability=approval_probability,
            rejection_probability=rejection_probability,
            selected_loan_type=loan_type,
            applicant_id=applicant_id,
            required_documents=required_documents
        )
    except (ValueError, KeyError) as e:
        return render_template("index.html", error_message=f"Please check the submitted form values. Details: {e}"), 400
    except Exception as e:
        app.logger.exception("Prediction failed")
        return render_template("index.html", error_message="Unable to generate a prediction. Check that the trained model is available and the form fields are valid."), 500

if __name__ == "__main__":
    app.run(debug=True)
