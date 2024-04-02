from flask import Blueprint, jsonify  # jsonify creates an endpoint response object
from flask_restful import Api, request, Resource # used for REST API building
import requests  # used for testing
import random
from __init__ import app, db
from model.realEstateModels import House, Favorite
# from ml.RealEstateRecEngine import RecEngine
from ai.OpenAIEngine import HouseAIEngine
import os
from cryptography.fernet import Fernet
from ml.realestatemlmodel import RecEngine
from ml.jobsalaryhousemodel import SalaryRecEngine

realestate_api = Blueprint('house', __name__, url_prefix='/api/house')


api = Api(realestate_api)


class houses:

    class _getHouses(Resource):
        def get(self):
            houses = db.session.query(House).all()
            print("Hi")
            return jsonify([house.few_details() for house in houses])
        
    class _gethousedetails(Resource):
        def get(self):
            house = db.session.query(House).filter(House.id == int(request.args.get("id"))).first()
            return jsonify(house.all_details())
        
    class _getOpenAIResponse(Resource):
        model = HouseAIEngine()
        
        def get(self):
            encrypted_code = 'gAAAAABlOL7JjQ7xkp55n2f2BkZ1KBaDu-bQgmbRJU7w6H25i2ImrdLwDcGPpD2YZejoTVAWSXWEM4vJLAB7r2YaXS9UAKOFjw=='
            crypto_key = os.getenv("CRYPTO_KEY")
            print(crypto_key)
            if crypto_key is None:
                raise ValueError("CRYPTO_KEY environment variable is not set.")
        
            cipher_suite = Fernet(crypto_key)
            decrypted_code = cipher_suite.decrypt(encrypted_code).decode()
            code = request.args.get("code")

            if code == decrypted_code:
                return jsonify(self.model.get_openai_answer(request.args.get("question")))
            else:
                return jsonify("UNAUTHORIZED")
    
    class _addToFavorites(Resource):
        def post(self):
            favoriteHouse = Favorite(account_id=request.args.get("id"), house_id=request.args.get("house_id")) 
            db.session.add(favoriteHouse)
            db.session.commit()
        
    class _getFavorites(Resource):
        def get(self):
            houses_id = set([favorite.house_id for favorite in db.session.query(Favorite).filter(Favorite.account_id == request.args.get("id")).all()])
            houses = [db.session.query(House).filter(House.id == house).first() for house in houses_id]
            return jsonify([house.few_details() for house in houses])
        
    class _MLPriceModel(Resource):
        model = RecEngine()

        def get(self):
            bed = int(request.args.get("bed"))
            bath = int(request.args.get("bath"))
            area = int(request.args.get("area"))
            rentestimate = int(request.args.get("rent"))
            return int(self.model.predictPrice(bath, bed, area, rentestimate))
        
    class _MLSalaryModel(Resource):
        model = SalaryRecEngine()

        def get(self):
            job_salary = int(request.args.get("salary"))
            predictedhouseprice = int(self.model.predictPrice(job_salary))
            houses = db.session.query(House).filter(House.price >= predictedhouseprice - 1000000, House.price <= predictedhouseprice + 1000000).all()
            if houses == []:
                return jsonify("No houses found")
            return jsonify([house.few_details() for house in houses])
                
    api.add_resource(_getHouses, "/houses")
    api.add_resource(_gethousedetails, "/housedetails")
    api.add_resource(_getOpenAIResponse, "/openai")
    api.add_resource(_addToFavorites, "/addtofavorites")
    api.add_resource(_getFavorites, "/getfavorites")
    api.add_resource(_MLPriceModel, "/mlpricemodel")
    api.add_resource(_MLSalaryModel, "/mlsalarymodel")