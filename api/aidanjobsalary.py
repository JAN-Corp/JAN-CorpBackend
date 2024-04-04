from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource # used for REST API building
from ml.aidanjobsalary import AidanSalaryModel
# Import the TitanicModel class from the model file
# from model.titanic import TitanicModel

AidanJobSalaryAPI = Blueprint('AidanJobSalaryAPI', __name__,
                   url_prefix='/api/aidanjobsalary')

api = Api(AidanJobSalaryAPI)
class TitanicAPI:
    class _Predict(Resource):
        
        def post(self):
            """ Semantics: In HTTP, POST requests are used to send data to the server for processing.
            Sending job_data data to the server to get a prediction fits the semantics of a POST request.
            
            POST requests send data in the body of the request...
            1. which can handle much larger amounts of data and data types, than URL parameters
            2. using an HTTPS request, the data is encrypted, making it more secure
            3. a JSON formated body is easy to read and write between JavaScript and Python, great for Postman testing
            """     
            # Get the job_data data from the request
            job_data = request.get_json()

            # Get the singleton instance of the TitanicModel
            aidanModel = AidanSalaryModel.get_instance()
            # Predict the survival probability of the job_data
            response = aidanModel.predictSalary(job_data)

            # Return the response as JSON
            return jsonify(response)

    api.add_resource(_Predict, '/predict')