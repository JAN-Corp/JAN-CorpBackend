""" database dependencies to support sqliteDB examples """
from random import randrange
from datetime import date
import os, base64
import json
from .aidanuser import AidanUser
from .jobs import Job
from .jobuser import JobUser
from __init__ import app, db
from sqlalchemy.exc import IntegrityError
from werkzeug.security import generate_password_hash, check_password_hash


''' Tutorial: https://www.sqlalchemy.org/library.html#tutorials, try to get into Python shell and follow along '''

# Define the Post class to manage actions in 'posts' table,  with a relationship to 'users' table

class Application(db.Model):
    __tablename__ = 'applications'

    # Define the Notes schema
    id = db.Column(db.Integer, primary_key=True)
    userid = db.Column(db.Integer, db.ForeignKey('aidansusers.id'))
    jobid = db.Column(db.Integer, db.ForeignKey('jobs.id'))
    address = db.Column(db.String(255), unique=False, nullable=False)
    email = db.Column(db.String(255), unique=False, nullable=False)
    separationFactor = db.Column(db.String(255), unique=False, nullable=False)

    # Constructor of a Notes object, initializes of instance variables within object
    def __init__(self, userid, jobid, address, email, separationFactor):
        self.userid = userid
        self.jobid = jobid
        self.address = address
        self.email = email
     
        self.separationFactor = separationFactor
        

    # Returns a string representation of the Notes object, similar to java toString()
    # returns string
    def update(self, address="", email="", separationFactor=""):
        """only updates values with length"""
        if len(address) > 0:
            self.address = address    # variables with self prefix become part of the object, 
        if len(email) > 0:
            self.email = email
        if len(separationFactor) > 0:
            self.separationFactor = separationFactor
        db.session.commit()
        return self


    # CRUD create, adds a new record to the Notes table
    # returns the object added or None in case of an error
    def create(self):
        try:
            # creates a Notes object from Notes(db.Model) class, passes initializers
            db.session.add(self)  # add prepares to persist person object to Notes table
            db.session.commit()  # SqlAlchemy "unit of work pattern" requires a manual commit
            return self
        except IntegrityError:
            db.session.remove()
            return None

    # CRUD read, returns dictionary representation of Notes object
    # returns dictionary
    def read(self):
    
        return {
            "id": self.id,
            "userid": self.userid,
            "jobid": self.jobid,
            "address": self.address,
            "email": self.email,
     
            "separationFactor": self.separationFactor
        }


def initApplications():
    with app.app_context():
        """Create database and tables"""
        db.create_all()
        """Tester data for table"""
     
   
        applications = [
            Application(userid = 1, jobid=1, address="16601 Nighthawk Ln", email="nighthawkcodingsociety@gmail.com", separationFactor="I am unique"),
            Application(userid = 1, jobid=1, address="16601 Nighthawk Ln", email="nighthawkcodingsociety@gmail.com", separationFactor="I am unique")
        ]
        for application in applications:
            try:
                application.create()
            except IntegrityError:
                '''fails with bad or duplicate data'''
                db.session.remove()
                print(f"Records exist, duplicate title, or error: {application.title}")