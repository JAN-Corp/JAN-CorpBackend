from flask import Flask, request, jsonify
from flask import Blueprint
from flask_restful import Api, Resource
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


class SalaryRecEngine:
    
    regressor = None

    def __init__(self):
        self.buildModel()
        
    def buildModel(self):
        basedir = os.path.abspath(os.path.dirname(__file__))

        # Specify the file path
        file_path = basedir + "/../static/data/RealEstateData.csv"

        data = pd.read_csv(file_path)
        data = data.dropna()
        #print(data)

        self.regressor = LinearRegression()
        X = data.drop(['price', 'address', 'city', 'state', 'zipcode', 'latitude', 'longitude', 'homeType', 'imgSrc', 'PriceEstimate', 'bathrooms', 'bedrooms', 'livingArea', 'RentEstimate'], axis=1)
        Y = data['price']
        
        X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.2, random_state=0)
        self.regressor.fit(X_Train, Y_Train)

        Y_Prediction = self.regressor.predict(X_Test)

        df = pd.DataFrame({'Actual ': Y_Test, 'Predicted': Y_Prediction})
        # print(df)

        mae = metrics.mean_absolute_error(Y_Test, Y_Prediction)
        r2 = metrics.r2_score(Y_Test, Y_Prediction)
        
        # print("The model performance for testing set")
        # print("-------------------------------------")
        # print('MAE is {}'.format(mae))
        # print('R2 score is {}'.format(r2))

    def predictPrice(self, JobSalary):
        predicion = self.regressor.predict([[JobSalary]])
        return predicion[0]


# Instantiate the RecEngine class
rec_engine = SalaryRecEngine()
print(rec_engine.predictPrice(1800000))
