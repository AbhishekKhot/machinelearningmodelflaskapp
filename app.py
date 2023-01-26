from flask import Flask, request, jsonify
import numpy as np
import sklearn
import pickle
import os
import joblib

scaler = pickle.load(open('scaler.pkl','rb'))
# dt = joblib.load(open('dt.sav'))
dt = joblib.load('dt.sav', mmap_mode=None)

app = Flask(__name__)


@app.route('/')
def index():
    return "Hello World"


@app.route('/result', methods=['POST'])
def result():
    gender = int(request.form['gender'])
    age = int(request.form['age'])
    hypertension = int(request.form['hypertension'])
    heart_disease = int(request.form['heart_disease'])
    ever_married = int(request.form['ever_married'])
    work_type = int(request.form['work_type'])
    Residence_type = int(request.form['Residence_type'])
    avg_glucose_level = float(request.form['avg_glucose_level'])
    bmi = float(request.form['bmi'])
    smoking_status = int(request.form['smoking_status'])

    x = np.array([gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type,
                  avg_glucose_level, bmi, smoking_status]).reshape(1, -1)

    x = scaler.transform(x)

    Y_pred = dt.predict(x)

    # for No Stroke Risk
    # if Y_pred == 0:
    #     return jsonify({'No chances': str(Y_pred)})
    # else:
    #     return jsonify({'Have chances': str(Y_pred)})

    return jsonify({'PredictionResult': str(Y_pred)})


# new changes has been done what the hell is this
if __name__ == '__main__':
    app.run(debug=True)