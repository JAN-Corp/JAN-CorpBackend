## Python Titanic Sample API endpoint
from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource # used for REST API building

# Import the TitanicModel class from the model file
from ml.jobsalary import SalaryRecEngine

houseJob_api = Blueprint('houseJob_api', __name__,
                   url_prefix='/api/houseJob')

api = Api(houseJob_api)
class jobSalaryAPI:
    class _Predict(Resource):
        model = SalaryRecEngine()

        def get(self):
            experience_level = int(request.args.get("experience_level"))
            employment_type = int(request.args.get("employment_type"))
            job_title = int(request.args.get("job_title"))
            work_setting = int(request.args.get("work_setting"))
            return int(self.model.predictSalary(experience_level, employment_type, job_title, work_setting))

    api.add_resource(_Predict, '/predict')