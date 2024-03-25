## Python Titanic Model

# Import the required libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import seaborn as sns

# Define the TitanicRegression global variable
titanic_regression = None

# Define the TitanicRegression class
class TitanicRegression:
    def __init__(self):
        self.dt = None
        self.logreg = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.encoder = None

    def initTitanic(self):
        titanic_data = sns.load_dataset('titanic')
        X = titanic_data.drop('survived', axis=1)
        y = titanic_data['survived']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Initialize the encoder
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        self.X_train = self.encoder.fit_transform(self.X_train)
        self.X_test = self.encoder.transform(self.X_test)

        self.dt = DecisionTreeClassifier()
        self.dt.fit(self.X_train, self.y_train)

        self.logreg = LogisticRegression()
        self.logreg.fit(self.X_train, self.y_train)

    def runDecisionTree(self):
        if self.dt is None:
            print("Decision Tree model is not initialized. Please run initTitanic() first.")
            return
        y_pred_dt = self.dt.predict(self.X_test)
        accuracy_dt = accuracy_score(self.y_test, y_pred_dt)
        print('Decision Tree Classifier Accuracy: {:.2%}'.format(accuracy_dt))

    def runLogisticRegression(self):
        if self.logreg is None:
            print("Logistic Regression model is not initialized. Please run initTitanic() first.")
            return
        y_pred_logreg = self.logreg.predict(self.X_test)
        accuracy_logreg = accuracy_score(self.y_test, y_pred_logreg)
        print('Logistic Regression Accuracy: {:.2%}'.format(accuracy_logreg))


def initTitanic():
    global titanic_regression
    titanic_regression = TitanicRegression()
    titanic_regression.initTitanic()
    titanic_regression.runDecisionTree()
    titanic_regression.runLogisticRegression()


def predictSurvival(passenger):
    passenger_df = pd.DataFrame(passenger, index=[0])
    passenger_df.drop(['name'], axis=1, inplace=True)

    # Encode categorical variables
    passenger_encoded = titanic_regression.encoder.transform(passenger_df)

    # Add missing columns and fill them with default values
    missing_cols = set(titanic_regression.encoder.get_feature_names_out()) - set(passenger_df.columns)
    for col in missing_cols:
        passenger_df[col] = 0

    # Ensure the order of columns in the passenger matches the order in the training data
    passenger_df = passenger_df[titanic_regression.X_train.columns]

    # Make prediction
    predict = titanic_regression.logreg.predict(passenger_encoded)

    return predict


# Sample usage
initTitanic()
passenger = {
        'name': ['John Mortensen'],
        'pclass': [2],
        'sex': ['male'],
        'age': [64],
        'sibsp': [1],
        'parch': [1],
        'fare': [16.00],
        'embarked': ['S'],
        'alone': [False]
    }
print(predictSurvival(passenger))