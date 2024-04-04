import json, jwt
from flask import Blueprint, request, jsonify, current_app, Response
from flask_restful import Api, Resource # used for REST API building
from datetime import datetime
from auth_middleware import token_required
from flask_cors import CORS
from model.users import Post
from sqlalchemy import func

post_api = Blueprint('post_api', __name__,
                   url_prefix='/api/post')
api = Api(post_api)



class PostAPI:        
    
    class _CRUD(Resource):  # User API operation for Create, Read.  THe Update, Delete methods need to be implemeented
        
          
        
        def post(self): # Create method
            ''' Read data for json body '''
            body = request.get_json()
            
            ''' Avoid garbage in, error checking '''
            # validate name
            note = body.get('note')
            if note is None or len(note) < 10:
                return {'message': f'Note is missing, or is less than 10 characters'}, 400
            # validate uid
            uid = body.get('uid')
            if uid is None or len(uid) < 2:
                return {'message': f'User ID is missing, or is less than 2 characters'}, 400
            # look for password and dob
            #doq = date of question
            doq = body.get('doq')
            parentPostId = body.get('parentPostId')
            ''' #1: Key code block, setup USER OBJECT '''
            uo = Post(id=uid, note= note, doq= doq, parentPostId=parentPostId )
                
            ''' Additional garbage error checking '''
            # convert to date type
            if doq is not None:
                try:
                    uo.doq = datetime.strptime(doq, '%Y-%m-%d').date()
                except:
                    return {'message': f'Date of birth format error {doq}, must be mm-dd-yyyy'}, 400
            
            ''' #2: Key Code block to add user to database '''
            # create user in database
            question = uo.create()
            # success returns json of user
            if question:
                return jsonify(question.read())
            # failure returns error
            return {'message': f'Processed {question}, either a format error or User ID {uid} is duplicate'}, 400

        def get(self):  # Read Method
            # Check if 'id' parameter is provided
            id = request.args.get('id')
            if id:
                # Query the post by id
                post = Post.query.filter_by(id=id).first()
                if post:
                    return jsonify(post.read())  # Assuming 'post.read()' method returns a JSON-serializable dict
                else:
                    return jsonify({"message": "No post found with the id " + str(id)})

            # Check if 'parentPostId' parameter is provided
            parentPostId = request.args.get('parentPostId')
            if parentPostId:
                # Query posts by parentPostId
                parent_posts = Post.query.filter_by(parentPostId=parentPostId).all()
                if parent_posts:
                    return jsonify([post.read() for post in parent_posts])
                else:
                    return jsonify({"message": "No posts found with the parentPostId " + str(parentPostId)})

            # If neither 'id' nor 'parentPostId' is provided, check for 'searchString'
            searchString = request.args.get('searchString')
            if searchString:
                filtered_posts = Post.query.filter(func.lower(func.trim(Post.note)).like('%' + searchString.lower().strip() + '%')).all()
                if not filtered_posts:
                    return jsonify({"message": "No posts found with the note " + searchString})
                return jsonify([post.read() for post in filtered_posts])

            # If none of the specific filters are provided, return all posts
            posts = Post.query.all()
            return jsonify([post.read() for post in posts])
       
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
                question = question.query.filter_by(_uid=uid).first()
                if question is None or not question.is_password(password):
                    return {'message': f"Invalid user id or password"}, 400
                if question:
                    try:
                        token = jwt.encode(
                            {"_uid": question._uid},
                            current_app.config["SECRET_KEY"],
                            algorithm="HS256"
                        )
                        resp = Response("Authentication for %s successful" % (question._uid))
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

            
    # building RESTapi endpoint
    # building RESTapi endpoint
    api.add_resource(_CRUD, '/')
    #api.add_resource(_Security, '/questions')
    