import json, jwt
from flask import Blueprint, request, jsonify, current_app, Response
from flask_restful import Api, Resource # used for REST API building
from datetime import datetime
from aidan_auth_middleware import aidan_token
from model.aidanuser import AidanUser
import random
from __init__ import app, db, cors
import flask
from model.jobs import Job
from model.jobuser import JobUser
from urllib import parse
from urllib.parse import urlparse
from urllib.parse import parse_qs
jobuser_api = Blueprint('jobuser_api', __name__,
                   url_prefix='/api/jobuser')

# API docs https://flask-restful.readthedocs.io/en/latest/api.html
api = Api(jobuser_api)



class JobUserAPI:        
    class _CRUD(Resource):  # User API operation for Create, Read.  THe Update, Delete methods need to be implemeented

        def post(self): # Create method
            ''' Read data for json body '''
            body = request.get_json()
            
            ''' Avoid garbage in, error checking '''
            # validate name
            jobid = body.get('jobid')
            userid = body.get('userid')
    
            ''' #1: Key code block, setup USER OBJECT '''
            juo = JobUser(jobid=jobid,
                          userid=userid)


            ''' #2: Key Code block to add user to database '''
            # create user in database
            jobuser = juo.create()
            # success returns json of user
            if jobuser:
                return jsonify(juo.read())

            # failure returns error
            return {'message': f'Processed {juo}, either a format error or User ID {userid} is duplicate'}, 400

   
        def get(self): # Read Method
            print(request.url)
            frontendrequest = request.url
            parsed_url = urlparse(frontendrequest)
            print("parsed_url")
            print(parsed_url)
            query_params = parse_qs(parsed_url.query)
            if 'id' in query_params:
                query_id = query_params['id'][0]
                print('query_id')
                print(query_id)
                job = Job.query.filter_by(id=query_id).first()
                if job:
                    return job.read()
                else:
                    return {'message': 'Job not found'}, 404
            else:
                jobs = Job.query.all()    # read/extract all users from database
                json_ready = [job.read() for job in jobs]  # prepare output in json
                return jsonify(json_ready) # jsonify creates Flask response object, more specific to APIs than json.dumps

                    
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
            
    
    class _ApplyCount(Resource):
        def get(self): # Read Method
            print(request.url)
            frontendrequest = request.url
            parsed_url = urlparse(frontendrequest)
            print("apply count")
            print(parsed_url)
            query_params = parse_qs(parsed_url.query)
            if 'id' in query_params:
                query_id = query_params['id'][0]
                print('apply count')
                print(query_id)
                count = JobUser.query.filter_by(jobid=query_id).count()
                print(count)
                if count:
                    return jsonify(count)
                else:
                    return jsonify('0')
            else:
                jobs = Job.query.all()    # read/extract all users from database
                json_ready = [job.read() for job in jobs]  # prepare output in json
                return jsonify(json_ready)
    class _Profile(Resource):
        def get(self):
            user = AidanUser.query.filter_by(id=request.args.get("userid")).first()

            if user is None:
                return jsonify({"error": "User not found"})
                
            if user.status == "Freelancer":
                jobs_id = set([jobuser.jobid for jobuser in db.session.query(JobUser).filter(JobUser.userid == request.args.get("userid")).all()])
                jobs = [db.session.query(Job).filter(Job.id == job).first() for job in jobs_id]
                return jsonify([job.read() for job in jobs])
            elif user.status == "Employer": # if user.status is employer
                jobpostee = Job.query.filter_by(_jobpostee=request.args.get("userid")).all()
                # get employer id, read all jobs they posted. posted from jobs
                return jsonify([job.read() for job in jobpostee])
            
    class _whoApplied(Resource):
        def get(self):
            users_id = set([jobuser.userid for jobuser in db.session.query(JobUser).filter(JobUser.jobid == request.args.get("id")).all()])
            users = [db.session.query(AidanUser).filter(AidanUser.id == user).first() for user in users_id]
            return jsonify([user.read() for user in users])
    
    class _userStatus(Resource):
        def get(self):
            user = AidanUser.query.filter_by(id=request.args.get("userid")).first()

            if user is None:
                return jsonify({"error": "User not found"})
                
            if user.status == "Freelancer":
                return {'status': 'Freelancer',
                    'name': f'{user._name}'}
             
            elif user.status == "Employer": # if user.status is employer
                return {'status': 'Employer',
                    'name': f'{user._name}'}
    

            
    # building RESTapi endpoint
    api.add_resource(_CRUD, '/')
    api.add_resource(_ApplyCount, '/applycount')
    api.add_resource(_Profile, '/profile')
    api.add_resource(_whoApplied, '/whoapplied')
    api.add_resource(_userStatus, '/userstatus')

    
    