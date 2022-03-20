from flask import Flask, request, jsonify
import pickle
import numpy as np

#load the model
model = pickle.load(open('Loan_ML_model.sav','rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return 'Hello World'

@app.route('/predict', methods =['POST'])
def predict():
    Gender = request.form.get('Gender')
    Married =  request.form.get('Married')
    Dependents =  request.form.get('Dependents')
    Education =  request.form.get('Education')
    Self_Employed =  request.form.get('Self_Employed')
    ApplicantIncome =  request.form.get('ApplicantIncome')
    LoanAmount =  request.form.get('LoanAmount')
    Loan_Amount_Term =  request.form.get('Loan_Amount_Term')
    Credit_History =  request.form.get('Credit_History')
    Property_Area =  request.form.get('Property_Area')
    
    input_query =  np.array([[Gender, Married, Dependents, Education, Self_Employed,ApplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area]])

    result = model.predict(input_query)[0]

    return jsonify({'Loan Status': str(result)})

if __name__ == '__main__' :
    app.run(debug = True)