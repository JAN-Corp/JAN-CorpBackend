# import threading

# import "packages" from flask
# import render_template from "public" flask libraries
from flask import render_template, request
from flask.cli import AppGroup

     
# import "packages" from "this" project
# from __init__ import app, db, cors  # Definitions initializatio
from __init__ import app, db, cors


# setup APIs
from model.realEstateModels import initHouses
from api.realestaste import realestate_api
from api.user import user_api  # Blueprint import api definition
from api.player import player_api
from api.titanic import titanic_api
from api.user import user_api # Blueprint import api definition
from api.post import post_api


# database migrations
from model.users import initUsers
from model.players import initPlayers

# setup App pages
# Blueprint directory import projects definition
from projects.projects import app_projects

# Initialize the SQLAlchemy object to work with the Flask app instance
db.init_app(app)

# register URIs
app.register_blueprint(user_api)  # register api routes
app.register_blueprint(player_api)
app.register_blueprint(app_projects)  # register app pages
app.register_blueprint(realestate_api)
app.register_blueprint(post_api)
app.register_blueprint(titanic_api)


with app.app_context():
    initHouses()
    initUsers()

@app.errorhandler(404)  # catch for URL not found
def page_not_found(e):
    # note that we set the 404 status explicitly
    return render_template('404.html'), 404


@app.route('/')  # connects default URL to index() function
def index():
    return render_template("index.html")


@app.route('/table/')  # connects /stub/ URL to stub() function
def table():
    return render_template("table.html")

'''
@app.before_request
def before_request():
    allowed_origin = request.headers.get('Origin')
    if allowed_origin in ['http://localhost:4100', 'http://127.0.0.1:4100', 'https://nighthawkcoders.github.io', 'https://real-estate-analyzation.github.io']:
        cors._origins = allowed_origin
    

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:8080')
    allowed_origins = ['http://localhost:4100', 'http://localhost:4200', 'http://127.0.0.1:4200', 'https://nighthawkcoders.github.io', 'https://real-estate-analyzation.github.io']

    origin = request.headers.get('Origin')
    if origin and origin in allowed_origins:
        response.headers.add('Access-Control-Allow-Origin', origin)

    response.headers.add('Access-Control-Allow-Origin', request.headers.get('Origin'))
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response
'''

# Create an AppGroup for custom commands
custom_cli = AppGroup('custom', help='Custom commands')

# Define a command to generate data


@custom_cli.command('generate_data')
def generate_data():
    initUsers()
    initPlayers()


# Register the custom command group with the Flask application
app.cli.add_command(custom_cli)

# this runs the application on the development server
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port="8181")
