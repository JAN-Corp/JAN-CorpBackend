from flask import Flask, request, jsonify, Blueprint
from flask_restful import Api, Resource
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Salary:
    
    def __init__(self):
        data = pd.read_csv('static/salarydata.csv')
        print('In init Salary')
        X = data[['YearsExperience']]
        y = data['Salary']
        print(data.head(10))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.logreg = LogisticRegression(max_iter=1000)
        self.logreg.fit(X_train_scaled, y_train)

    def predict(self, data):
        try:
            input_data = pd.DataFrame([data])
            scaler = StandardScaler()
            input_data_scaled = scaler.fit_transform(input_data)

            prediction = self.logreg.predict(input_data_scaled)

            return {'Prediction': int(prediction[0])}

        except Exception as e:
            return {'error': str(e)}
        



