import json, jwt
from flask import Blueprint, request, jsonify, current_app, Response
from flask_restful import Api, Resource # used for REST API building
from datetime import datetime
from aidan_auth_middleware import aidan_token
from model.aidanuser import AidanUser
import random
from __init__ import app, db, cors
import flask

aidanuser_api = Blueprint('aidanuser_api', __name__,
                   url_prefix='/api/aidanuser')

# API docs https://flask-restful.readthedocs.io/en/latest/api.html
api = Api(aidanuser_api)




class UserAPI:        
    class _CRUD(Resource):  # User API operation for Create, Read.  THe Update, Delete methods need to be implemeented
        #@token_required()
        def post(self): # Create method
            ''' Read data for json body '''
            body = request.get_json()
            
            ''' Avoid garbage in, error checking '''
            # validate name
            name = body.get('name')
            status = body.get('status')
            if name is None or len(name) < 2:
                return {'message': f'Name is missing, or is less than 2 characters'}, 400
            # validate uid
            #uid = body.get('uid')
            #if uid is None or len(uid) < 2:
            #    return {'message': f'User ID is missing, or is less than 2 characters'}, 400
            # look for password and dob
            role = body.get('role')
            password = body.get('password')
            dob = body.get('dob')
            if dob is not None:
                try:
                    dob = datetime.strptime(dob, '%Y-%m-%d').date()
                except:
                    return {'message': f'Date of birth format error {dob}, must be yyyy-mm-dd'}, 400

            UserName = body.get('name')[:4]
            UserId = random.randrange(100, 999)
            NewUserId = f"{UserName}" + f"{UserId}"
          
            
            ''' #1: Key code block, setup USER OBJECT '''
            uo = AidanUser(name=name, 
                      uid=NewUserId,
                      dob=dob,
                      status=status,
                      role=role)

            
            ''' Additional garbage error checking '''
            # set password if provided
            if password is not None:
                uo.set_password(password)
            # convert to date type
            
            
            ''' #2: Key Code block to add user to database '''
            # create user in database
            user = uo.create()
            # success returns json of user
            if user:
                return jsonify(NewUserId)

            # failure returns error
            return {'message': f'Processed {name}, either a format error or User ID {NewUserId} is duplicate'}, 400

        @aidan_token("Employer")
        def get(self, current_user): # Read Method
            users = AidanUser.query.all()    # read/extract all users from database
            json_ready = [user.read() for user in users]  # prepare output in json
            return jsonify(json_ready)  # jsonify creates Flask response object, more specific to APIs than json.dumps
        
        @aidan_token("Employer")
        def delete(self, _): # Delete Method
            body = request.get_json()
            uid = body.get('uid')
            user = AidanUser.query.filter_by(_uid=uid).first()
            if user is None:
                return {'message': f'User {uid} not found'}, 404
            json = user.read()
            user.delete() 
            # 204 is the status code for delete with no json response
            return f"Deleted user: {json}", 204 # use 200 to test with Postman
            
    
    class _Security(Resource):
        def post(self):
            try:
                body = request.get_json()
                if not body:
                    return {
                        "message": "Please provide user details",
                        "data": None,
                        "error": "Bad request"
                    }, 400
                ''' Get Data '''
                uid = body.get('uid')
                if uid is None:
                    return {'message': f'User ID is missing'}, 400
                password = body.get('password')
                
                ''' Find user '''
                user = AidanUser.query.filter_by(_uid=uid).first()
                if user is None or not user.is_password(password):
                    return {'message': f"Invalid user id or password"}, 400
                if user:
                    try:
                        token = jwt.encode(
                            {"_uid": user._uid,
                            "role": user.role},
                            current_app.config["SECRET_KEY"],
                            algorithm="HS256"
                        )
                        resp = jsonify(user.id)
                        resp.set_cookie("jwt", token,
                                max_age=3600,
                                secure=True,
                                httponly=True,
                                path='/',
                                samesite='None'  # This is the key part for cross-site requests

                                # domain="frontend.com"
                                )
                   
                        return resp
                    except Exception as e:
                        return {
                            "error": "Something went wrong",
                            "message": str(e)
                        }, 500
                return {
                    "message": "Error fetching auth token!",
                    "data": None,
                    "error": "Unauthorized"
                }, 404
            except Exception as e:
                return {
                        "message": "Something went wrong!",
                        "error": str(e),
                        "data": None
                }, 500
    class _Apply(Resource):
        @aidan_token('Freelancer')
        def get(self, current_user): # Read Method
            users = AidanUser.query.all()    # read/extract all users from database
            json_ready = [user.read() for user in users]  # prepare output in json
            return jsonify(json_ready)  # jsonify creates Flask response object, more specific to APIs than json.dumps

            
    # building RESTapi endpoint
    api.add_resource(_CRUD, '/')
    api.add_resource(_Security, '/authenticate')
    api.add_resource(_Apply, '/apply')
    