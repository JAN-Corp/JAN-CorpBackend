from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import seaborn as sns
import os

class AidanSalaryModel:
    """A class used to represent the Titanic Model for passenger survival prediction.
    """
    # a singleton instance of TitanicModel, created to train the model only once, while using it for prediction multiple times
    _instance = None
    
    # constructor, used to initialize the TitanicModel
    def __init__(self):
        # the titanic ML model
        self.model = None
        self.dt = None
        # define ML features and target
        self.features = ['employment_type', 'work_setting']
        self.target = 'salary_in_usd'
        # load the titanic dataset
        basedir = os.path.abspath(os.path.dirname(__file__))
        file_path = basedir + "/../static/data/jobs.csv"
        self.jobsalary_data = pd.read_csv(file_path)
     
        # one-hot encoder used to encode 'embarked' column
        self.encoder = OneHotEncoder(handle_unknown='ignore')

    # clean the titanic dataset, prepare it for training
    def _clean(self):
        # Drop unnecessary columns
        self.jobsalary_data.drop(['experience_level', 'work_year','job_title','salary','salary_currency','employee_residence','company_location','company_size','job_category'], axis=1, inplace=True)

        # Convert boolean columns to integers
        self.jobsalary_data['employment_type'] = self.jobsalary_data['employment_type'].apply(lambda x: 1 if x == 'Full-time' else 0)
        self.jobsalary_data['work_setting'] = self.jobsalary_data['work_setting'].apply(lambda x: 1 if x == 'In-person' else 0)

        # Drop rows with missing 'embarked' values before one-hot encoding
        # self.jobsalary_data.dropna(subset=['experience_level'], inplace=True)
        
        # One-hot encode 'embarked' column
        # onehot = self.encoder.fit_transform(self.jobsalary_data[['experience_level']]).toarray()
        # cols = ['experience_level_' + str(val) for val in self.encoder.categories_[0]]
        # onehot_df = pd.DataFrame(onehot, columns=cols)
        # self.jobsalary_data = pd.concat([self.jobsalary_data, onehot_df], axis=1)
        # self.jobsalary_data.drop(['experience_level'], axis=1, inplace=True)

        # Add the one-hot encoded 'embarked' features to the features list
       #self.features.extend(cols)
        
        # Drop rows with missing values
        self.jobsalary_data.dropna(inplace=True)

    # train the titanic model, using logistic regression as key model, and decision tree to show feature importance
    def _train(self):
        # split the data into features and target
        X = self.jobsalary_data[self.features]
        y = self.jobsalary_data[self.target]
        
        # perform train-test split
        self.model = LogisticRegression(max_iter=1000)
        
        # train the model
        self.model.fit(X, y)
        
        # train a decision tree classifier
        self.dt = DecisionTreeClassifier()
        self.dt.fit(X, y)
        
    @classmethod
    def get_instance(cls):
        """ Gets, and conditionaly cleans and builds, the singleton instance of the TitanicModel.
        The model is used for analysis on titanic data and predictions on the survival of theoritical passengers.
        
        Returns:
            TitanicModel: the singleton _instance of the TitanicModel, which contains data and methods for prediction.
        """        
        # check for instance, if it doesn't exist, create it
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._clean()
            cls._instance._train()
        # return the instance, to be used for prediction
        return cls._instance

    def predictSalary(self, job_data):
        # Prepare the job data for prediction
        job_df = pd.DataFrame(job_data, index=[0])
        job_df['employment_type'] = job_df['employment_type'].apply(lambda x: 1 if x == 'Full-time' else 0)
        job_df['work_setting'] = job_df['work_setting'].apply(lambda x: 1 if x == 'In-person' else 0)
            
        # onehot = self.encoder.transform(job_df[['experience_level']]).toarray()
        # cols = ['experience_level_' + str(val) for val in self.encoder.categories_[0]]
        # onehot_df = pd.DataFrame(onehot, columns=cols)
        # job_df = pd.concat([job_df, onehot_df], axis=1)
        
        # Now drop 'experience_level' column after concatenating
        job_df.drop(['experience_level'], axis=1, inplace=True)
            
        # Predict the salary
        predicted_salary = self.model.predict([job_df])

        # Return the predicted salary
        return predicted_salary[0]
    
    def feature_weights(self):
        """Get the feature weights
        The weights represent the relative importance of each feature in the prediction model.

        Returns:
            dictionary: contains each feature as a key and its weight of importance as a value
        """
        # extract the feature importances from the decision tree model
        importances = self.dt.feature_importances_
        # return the feature importances as a dictionary, using dictionary comprehension
        return {feature: importance for feature, importance in zip(self.features, importances)} 
    
def initJobSalary():
    """ Initialize the Titanic Model.
    This function is used to load the Titanic Model into memory, and prepare it for prediction.
    """
    AidanSalaryModel.get_instance()
    
