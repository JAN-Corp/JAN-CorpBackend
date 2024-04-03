from flask import Flask, request, jsonify, Blueprint
from flask_restful import Api, Resource
from model.Salary import Salary

salary_api = Blueprint('salary_api', __name__, url_prefix='/api/salary')
api = Api(salary_api)


class PredictOutcome(Resource):
    def post(self):
        try:
            body = request.get_json()
            print(f'in post: {body}') 
            salary = Salary()
            prediction = salary.predict(body)
            return prediction
        except Exception as e:
            return jsonify({'error': str(e)})

api.add_resource(PredictOutcome, '/predict')

app = Flask(__name__)
app.register_blueprint(salary_api)

if __name__ == '__main__':
    app.run(debug=True)
