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

        file_path = basedir + "/../static/data/jobs.csv"

        data = pd.read_csv(file_path)
        data = data.dropna()
        #print(data)

        features = ['experience_level', 'employment_type', 'job_title', 'work_setting']
        encoder = OneHotEncoder(drop='first')
        encoded_features = encoder.fit_transform(data[features]).toarray()
        encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(features))
        data = pd.concat([data.drop(columns=features), encoded_df], axis=1)


        X = data.drop(columns=['salary_in_usd'])
        y = data['salary_in_usd']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Step 4: Train a linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)


        # train_score = model.score(X_train, y_train)
        # test_score = model.score(X_test, y_test)
        # print("Train R2 Score:", train_score)
        # print("Test R2 Score:", test_score)

  
    
    def predictSalary(self, JobSalary):
        predicion = self.model.predict([[JobSalary]])
        return predicion[0]


# Instantiate the RecEngine class
rec_engine = SalaryRecEngine()
print(rec_engine.predictSalary(1800000))
