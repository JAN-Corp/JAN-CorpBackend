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
from model.applications import Application
from model.jobuser import JobUser
from urllib import parse
from urllib.parse import urlparse
from urllib.parse import parse_qs



job_api = Blueprint('job_api', __name__,
                   url_prefix='/api/job')

...
# API docs https://flask-restful.readthedocs.io/en/latest/api.html
api = Api(job_api)


class JobAPI:        
    class _CRUD(Resource):  # User API operation for Create, Read.  THe Update, Delete methods need to be implemeented
      
        def post(self): # Create method
            ''' Read data for json body '''
            body = request.get_json()
            
            ''' Avoid garbage in, error checking '''
            # validate name
            title = body.get('title')
            description = body.get('description')
            qualification = body.get('qualification')
            pay = body.get('pay')
            field = body.get('field')
            location = body.get('location')
            jobpostee = body.get('jobpostee')
    
            ''' #1: Key code block, setup USER OBJECT '''
            jo = Job(title=title, 
                      description=description,
                      qualification=qualification,
                      pay=pay,
                      field=field,
                      location=location,
                      jobpostee=jobpostee)

        

            ''' #2: Key Code block to add user to database '''
            # create user in database
            job = jo.create()
            # success returns json of user
            if job:
                return jsonify(title)

            # failure returns error
            return {'message': f'Processed {title}, either a format error or User ID {description} is duplicate'}, 400

        # this method is when users click on specific job(say "IT Help"), and it will return information about it
        def get(self): # Read Method
            
            
            frontendrequest = request.url
            parsed_url = urlparse(frontendrequest)
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



    class _updateJob(Resource):
        def put(self):
            job = Job.query.filter_by(id=request.args.get("id")).first()
            
            body = request.get_json()
            title = body.get('title')
            description = body.get('description')
            qualification = body.get('qualification')
            pay = body.get('pay')
            field = body.get('field')
            location = body.get('location')

            
            updatedJob = job.update(title=title, description=description,qualification=qualification
                                    ,pay=pay, field=field,location=location,   )


            # success returns json of user
            if updatedJob:
                return jsonify(job.read())

            # failure returns error
            return {'message': f'Processed {updatedJob}, format error'}, 400

    class _viewApplication(Resource):
        def get(self):
            application = Application.query.filter_by(jobid=request.args.get('jobid'), userid=request.args.get('userid')).first()

            return jsonify(application.read())                                             
    class _editApplication(Resource):
        def put(self):
            application = Application.query.filter_by(jobid=request.args.get('jobid'), userid=request.args.get('userid')).first()
            
            body = request.get_json()
            address = body.get('address')
            email = body.get('email')
            separationFactor = body.get('separationFactor')
        

            
            updatedApplication = application.update(address=address, email=email,separationFactor=separationFactor
                                    )


            # success returns json of user
            if updatedApplication:
                return jsonify(application.read())

            # failure returns error
            return {'message': f'Processed {updatedApplication}, format error'}, 400
    class _submitApplication(Resource):
        def post(self):
            body = request.get_json()
            
            ''' Avoid garbage in, error checking '''
            # validate name
            address = body.get('address')
            email = body.get('email')
            separationFactor = body.get('separationFactor')
            jobid = request.args.get('jobid')
            userid = request.args.get('userid')
    
            ''' #1: Key code block, setup USER OBJECT '''
            ao = Application(userid=userid,
                             jobid=jobid,
                             address=address,
                             email=email,
                             separationFactor=separationFactor)


            juo = JobUser(jobid=jobid,
                          userid=userid)
            ''' #2: Key Code block to add user to database '''
            # create user in database
            juo.create()
            application = ao.create()
            # success returns json of user
            if application:
                return jsonify(ao.read())

            # failure returns error
            return {'message': f'Processed {email}, either a format error or User ID {address} is duplicate'}, 400

        
    # building RESTapi endpoint
    api.add_resource(_CRUD, '/')
    api.add_resource(_updateJob, '/updatejob')
    api.add_resource(_viewApplication, '/viewapplication')
    api.add_resource(_editApplication, '/editapplication')
    api.add_resource(_submitApplication, '/submitapplication')
